/*
 * pluto_airband_scan.c
 *
 * VHF civil airband activity scanner for Pluto+/PlutoSDR-class libiio devices.
 *
 * This is intentionally structured like the Pluto-enabled dump1090.c template:
 *   - global Modes state
 *   - MODES_* defaults
 *   - modesInitConfig(), modesInit(), modesInitPLUTOSDR()
 *   - libiio access to cf-ad9361-lpc / ad9361-phy / voltage0+voltage1
 *
 * It does NOT decode aviation AM voice. It reports likely active channels by
 * looking for AM carrier / narrowband energy in the 118.000-136.975 MHz band.
 *
 * Build on the Pluto+/Pluto Linux shell:
 *   gcc -O2 -Wall -Wextra -o airband_scan pluto_airband_scan_v8.c -liio -lm
 *
 * v5 note: adds ACTIVE/ENDED hysteresis, duration tracking, summaries,
 * CSV/JSON logging, frequency watch-list files, priority revisits, and
 * split verbose/debug modes.
 *
 * v6 note: adds cleaner shutdown/release handling for Ctrl-C, SIGTERM,
 * SIGHUP, error exits, IIO buffer destruction, channel disable, and
 * heap/log cleanup.
 *
 * v7 note: adds --chirp export for CHIRP Generic CSV memory import.
 *
 * v8 note: fixes CHIRP CSV row width so data rows match the 18-column header.
 *
 * Example:
 *   ./airband_scan --threshold 16 --gain -100
 *   ./airband_scan --start 118000000 --end 136975000 --step 25000
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <iio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MODES_DEFAULT_RATE        2400000LL
#define MODES_DEFAULT_FREQ        1090000000LL
#define MODES_ASYNC_BUF_NUMBER    12
#define MODES_DATA_LEN            (16*16384)
#define MODES_AUTO_GAIN           -100.0   /* Use AGC */
#define MODES_MAX_GAIN            70.0
#define MODES_NOTUSED(V)          ((void)(V))

#define AIRBAND_START_HZ          118000000LL
#define AIRBAND_END_HZ            136975000LL
#define AIRBAND_STEP_HZ           25000LL
#define DEFAULT_FFT_SIZE          8192
#define DEFAULT_FFT_AVG           4
#define DEFAULT_USABLE_FRACTION   0.80
#define DEFAULT_THRESHOLD_DB      16.0
#define DEFAULT_DETECT_BW_HZ      12000.0
#define DEFAULT_REPORT_HOLDOFF_MS 5000LL /* retained for old CLI compatibility */
#define DEFAULT_SETTLE_MS         15
#define DEFAULT_ACTIVE_HITS       2
#define DEFAULT_INACTIVE_MISSES   3
#define DEFAULT_SUMMARY_TOP       10
#define DEFAULT_PRIORITY_RECENT_MS 30000LL
#define DEFAULT_PRIORITY_MAX      16

typedef struct {
    float re;
    float im;
} cpxf;

struct air_channel {
    long long freq;
    char label[96];

    /* Last measured state */
    double last_score_db;
    long long last_tested_ms;
    long long last_seen_ms;

    /* Hysteresis state */
    int active;
    int hit_streak;
    int miss_streak;
    long long first_hit_ms;
    long long active_start_ms;

    /* Counters and summary state */
    unsigned long hit_count;
    unsigned long miss_count;
    unsigned long activation_count;
    long long total_active_ms;
    double peak_score_db;

    /* CHIRP export state */
    int chirp_exported;
};

/* Program global state, kept similar to dump1090.c where practical. */
static struct {
    /* Internal state */
    unsigned char *data;       /* Raw IQ samples buffer, retained from template */
    uint16_t *magnitude;       /* Magnitude vector, retained from template */
    uint32_t data_len;
    uint16_t *maglut;          /* I/Q -> magnitude lookup table */
    int exit;

    /* PlutoSDR / libiio */
    int dev_index;
    double gain;
    int enable_agc;
    struct iio_context *ctx;
    struct iio_device *dev;
    struct iio_device *phy;
    struct iio_channel *rx0_phy;
    struct iio_channel *lo_chn;
    long long freq;
    long long sample_rate;
    long long rf_bandwidth;
    struct iio_channel *rx0_i;
    struct iio_channel *rx0_q;
    struct iio_buffer *rxbuf;
    int stop;

    /* Airband scanner config */
    long long scan_start;
    long long scan_end;
    long long channel_step;
    double threshold_db;
    double detect_bw_hz;
    double usable_fraction;
    int fft_size;
    int fft_avg;
    int settle_ms;
    long long report_holdoff_ms;
    int continuous;
    int verbose;
    int debug_scores;
    int debug_iio;

    /* Hysteresis / stateful reporting */
    int active_hits;
    int inactive_misses;

    /* Summary output */
    long long summary_every_ms;
    int summary_top;
    long long last_summary_ms;

    /* Event logging */
    int json_output;
    const char *csv_log_path;
    FILE *csv_log;

    /* CHIRP Generic CSV export.  This is a memory-channel file, not
     * a raw event log, so each detected frequency is exported once.
     */
    const char *chirp_csv_path;
    FILE *chirp_csv;
    int chirp_start_location;
    int chirp_next_location;
    int chirp_export_count;

    /* Optional watch-list and priority scanning */
    const char *freq_file;
    int scan_start_set;
    int scan_end_set;
    int priority_mode;
    long long priority_recent_ms;
    int priority_max_channels;

    struct air_channel *channels;
    int channel_count;
} Modes;

static volatile sig_atomic_t ShutdownSignal = 0;

static void requestShutdown(int sig) {
    /* Keep the signal handler async-signal-safe: only set simple flags.
     * The actual libiio cleanup is done later from the main thread.
     */
    ShutdownSignal = sig;
    Modes.exit = 1;
    Modes.stop = 1;
}

static void installSignalHandlers(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = requestShutdown;
    sigemptyset(&sa.sa_mask);

    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
#ifdef SIGHUP
    sigaction(SIGHUP, &sa, NULL);
#endif
}

static long long mstime(void) {
    struct timeval tv;
    long long mst;
    gettimeofday(&tv, NULL);
    mst = ((long long)tv.tv_sec) * 1000;
    mst += tv.tv_usec / 1000;
    return mst;
}

static void formatTimeMs(long long ms, char *buf, size_t len) {
    time_t sec = (time_t)(ms / 1000);
    struct tm tm_now;
    localtime_r(&sec, &tm_now);
    strftime(buf, len, "%Y-%m-%d %H:%M:%S", &tm_now);
}

static void formatDurationMs(long long ms, char *buf, size_t len) {
    long long total = ms / 1000;
    long long h = total / 3600;
    long long m = (total % 3600) / 60;
    long long s = total % 60;
    if (h > 99) {
        snprintf(buf, len, "%lld:%02lld:%02lld", h, m, s);
    } else {
        snprintf(buf, len, "%02lld:%02lld:%02lld", h, m, s);
    }
}

static char *trimWhitespace(char *s) {
    char *end;
    while (*s && isspace((unsigned char)*s)) s++;
    if (*s == '\0') return s;
    end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) *end-- = '\0';
    return s;
}

static void jsonPrintEscaped(FILE *fp, const char *s) {
    fputc('"', fp);
    for (; s && *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
            case '"': fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (c < 0x20) fprintf(fp, "\\u%04x", c);
                else fputc(c, fp);
        }
    }
    fputc('"', fp);
}

static void csvPrintEscaped(FILE *fp, const char *s) {
    fputc('"', fp);
    for (; s && *s; s++) {
        if (*s == '"') fputc('"', fp);
        fputc(*s, fp);
    }
    fputc('"', fp);
}

static int is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static void die_iio(const char *what, int err) {
    if (err < 0) {
        fprintf(stderr, "%s: %s (%d)\n", what, strerror(-err), err);
    } else {
        fprintf(stderr, "%s: %s\n", what, strerror(errno));
    }
    exit(1);
}

static void modesInitConfig(void) {
    Modes.gain = MODES_AUTO_GAIN;
    Modes.dev_index = 0;
    Modes.enable_agc = 1;
    Modes.freq = MODES_DEFAULT_FREQ;
    Modes.sample_rate = MODES_DEFAULT_RATE;
    Modes.rf_bandwidth = MODES_DEFAULT_RATE;

    Modes.scan_start = AIRBAND_START_HZ;
    Modes.scan_end = AIRBAND_END_HZ;
    Modes.channel_step = AIRBAND_STEP_HZ;
    Modes.threshold_db = DEFAULT_THRESHOLD_DB;
    Modes.detect_bw_hz = DEFAULT_DETECT_BW_HZ;
    Modes.usable_fraction = DEFAULT_USABLE_FRACTION;
    Modes.fft_size = DEFAULT_FFT_SIZE;
    Modes.fft_avg = DEFAULT_FFT_AVG;
    Modes.settle_ms = DEFAULT_SETTLE_MS;
    Modes.report_holdoff_ms = DEFAULT_REPORT_HOLDOFF_MS;
    Modes.continuous = 1;
    Modes.verbose = 0;
    Modes.debug_scores = 0;
    Modes.debug_iio = 0;

    Modes.active_hits = DEFAULT_ACTIVE_HITS;
    Modes.inactive_misses = DEFAULT_INACTIVE_MISSES;

    Modes.summary_every_ms = 0;
    Modes.summary_top = DEFAULT_SUMMARY_TOP;
    Modes.last_summary_ms = 0;

    Modes.json_output = 0;
    Modes.csv_log_path = NULL;
    Modes.csv_log = NULL;

    Modes.chirp_csv_path = NULL;
    Modes.chirp_csv = NULL;
    Modes.chirp_start_location = 1;
    Modes.chirp_next_location = 1;
    Modes.chirp_export_count = 0;

    Modes.freq_file = NULL;
    Modes.scan_start_set = 0;
    Modes.scan_end_set = 0;
    Modes.priority_mode = 0;
    Modes.priority_recent_ms = DEFAULT_PRIORITY_RECENT_MS;
    Modes.priority_max_channels = DEFAULT_PRIORITY_MAX;

    Modes.exit = 0;
    Modes.stop = 0;
}

static void modesInit(void) {
    int i, q;
    Modes.data_len = MODES_DATA_LEN;

    Modes.data = malloc(Modes.data_len);
    Modes.magnitude = malloc((Modes.data_len / 2) * sizeof(uint16_t));
    Modes.maglut = malloc(129 * 129 * sizeof(uint16_t));
    if (!Modes.data || !Modes.magnitude || !Modes.maglut) {
        fprintf(stderr, "Out of memory allocating buffers.\n");
        exit(1);
    }
    memset(Modes.data, 127, Modes.data_len);

    /* Same idea as dump1090.c: fast I/Q -> magnitude lookup. */
    for (i = 0; i <= 128; i++) {
        for (q = 0; q <= 128; q++) {
            Modes.maglut[i * 129 + q] = (uint16_t)round(sqrt((double)i*i + (double)q*q) * 360.0);
        }
    }
}

/* Retained from dump1090.c style. The scanner uses FFT power, but this is useful
 * for quick raw-envelope experiments and keeps the template structure familiar. */
static void computeMagnitudeVector(void) {
    uint16_t *m = Modes.magnitude;
    unsigned char *p = Modes.data;
    uint32_t j;
    for (j = 0; j + 1 < Modes.data_len; j += 2) {
        int i = p[j] - 127;
        int q = p[j + 1] - 127;
        if (i < 0) i = -i;
        if (q < 0) q = -q;
        if (i > 128) i = 128;
        if (q > 128) q = 128;
        m[j / 2] = Modes.maglut[i * 129 + q];
    }
}

static void modesCleanup(void) {
    static int cleaned = 0;
    if (cleaned) return;
    cleaned = 1;

    if (Modes.rxbuf) {
        if (Modes.verbose || Modes.debug_iio) fprintf(stderr, "* Destroying IIO RX buffer\n");
        iio_buffer_destroy(Modes.rxbuf);
        Modes.rxbuf = NULL;
    }

    /* Disable streaming channels after the buffer is gone. This is not
     * strictly required by libiio context destruction, but it makes the
     * release explicit and mirrors the normal enable/create order.
     */
    if (Modes.rx0_i) {
        if (Modes.verbose || Modes.debug_iio) fprintf(stderr, "* Disabling RX I channel\n");
        iio_channel_disable(Modes.rx0_i);
        Modes.rx0_i = NULL;
    }
    if (Modes.rx0_q) {
        if (Modes.verbose || Modes.debug_iio) fprintf(stderr, "* Disabling RX Q channel\n");
        iio_channel_disable(Modes.rx0_q);
        Modes.rx0_q = NULL;
    }

    if (Modes.ctx) {
        if (Modes.verbose || Modes.debug_iio) fprintf(stderr, "* Destroying IIO context\n");
        iio_context_destroy(Modes.ctx);
        Modes.ctx = NULL;
        Modes.dev = NULL;
        Modes.phy = NULL;
        Modes.rx0_phy = NULL;
        Modes.lo_chn = NULL;
    }

    if (Modes.csv_log) {
        fclose(Modes.csv_log);
        Modes.csv_log = NULL;
    }

    if (Modes.chirp_csv) {
        fclose(Modes.chirp_csv);
        Modes.chirp_csv = NULL;
    }

    free(Modes.channels);
    Modes.channels = NULL;
    Modes.channel_count = 0;

    free(Modes.maglut);
    Modes.maglut = NULL;

    free(Modes.magnitude);
    Modes.magnitude = NULL;

    free(Modes.data);
    Modes.data = NULL;
}

static void attr_write_ll(struct iio_channel *chn, const char *attr, long long val) {
    int ret = iio_channel_attr_write_longlong(chn, attr, val);
    if (ret < 0) die_iio(attr, ret);
}

static void attr_write_str(struct iio_channel *chn, const char *attr, const char *val) {
    int ret = iio_channel_attr_write(chn, attr, val);
    if (ret < 0) die_iio(attr, ret);
}

static long long attr_read_ll_or_default(struct iio_channel *chn, const char *attr, long long defval) {
    long long val = defval;
    int ret = iio_channel_attr_read_longlong(chn, attr, &val);
    if (ret < 0) return defval;
    return val;
}

static int attr_read_str(struct iio_channel *chn, const char *attr, char *buf, size_t len) {
    int ret = iio_channel_attr_read(chn, attr, buf, len);
    if (ret < 0) return ret;
    if (len > 0) buf[len - 1] = '\0';
    return ret;
}

static int parse_available_range(const char *s, long long *minv, long long *stepv, long long *maxv) {
    /* Typical format: "[2083333 1 61440000]" */
    const char *p = strchr(s, '[');
    char *end = NULL;
    long long a, b, c;
    if (!p) p = s; else p++;
    errno = 0;
    a = strtoll(p, &end, 10);
    if (errno || end == p) return -1;
    p = end;
    b = strtoll(p, &end, 10);
    if (errno || end == p) return -1;
    p = end;
    c = strtoll(p, &end, 10);
    if (errno || end == p) return -1;
    *minv = a;
    *stepv = b;
    *maxv = c;
    return 0;
}

static long long clamp_to_available_rate(struct iio_channel *chn, long long requested) {
    char avail[256];
    long long minv = 0, stepv = 1, maxv = 0;
    int ret = attr_read_str(chn, "sampling_frequency_available", avail, sizeof(avail));
    if (ret < 0) {
        fprintf(stderr, "Warning: could not read sampling_frequency_available; using requested rate %lld Hz.\n",
                requested);
        return requested;
    }
    if (parse_available_range(avail, &minv, &stepv, &maxv) < 0 || minv <= 0 || maxv <= 0) {
        fprintf(stderr, "Warning: could not parse sampling_frequency_available='%s'; using requested rate %lld Hz.\n",
                avail, requested);
        return requested;
    }

    long long adjusted = requested;
    if (adjusted < minv) adjusted = minv;
    if (adjusted > maxv) adjusted = maxv;
    if (stepv > 1 && adjusted > minv) {
        adjusted = minv + ((adjusted - minv + stepv - 1) / stepv) * stepv;
        if (adjusted > maxv) adjusted = maxv;
    }
    if (adjusted != requested) {
        fprintf(stderr,
                "Requested sample rate %lld Hz is outside this Pluto's available range %s; using %lld Hz.\n",
                requested, avail, adjusted);
    }
    return adjusted;
}

static long long clamp_rf_bandwidth(struct iio_channel *chn, long long requested) {
    char avail[256];
    long long minv = 0, stepv = 1, maxv = 0;
    int ret = attr_read_str(chn, "rf_bandwidth_available", avail, sizeof(avail));
    if (ret < 0) return requested;
    if (parse_available_range(avail, &minv, &stepv, &maxv) < 0 || minv <= 0 || maxv <= 0) return requested;

    long long adjusted = requested;
    if (adjusted < minv) adjusted = minv;
    if (adjusted > maxv) adjusted = maxv;
    if (stepv > 1 && adjusted > minv) {
        adjusted = minv + ((adjusted - minv + stepv - 1) / stepv) * stepv;
        if (adjusted > maxv) adjusted = maxv;
    }
    if (adjusted != requested) {
        fprintf(stderr,
                "Requested RF bandwidth %lld Hz is outside this Pluto's available range %s; using %lld Hz.\n",
                requested, avail, adjusted);
    }
    return adjusted;
}

static long long setCenterFrequency(long long center_hz) {
    attr_write_ll(Modes.lo_chn, "frequency", center_hz);

    /* Read back the LO frequency for verbose output.  Some firmware may round
     * the requested value slightly; showing the readback is more useful than
     * only showing the requested value.
     */
    Modes.freq = attr_read_ll_or_default(Modes.lo_chn, "frequency", center_hz);

    if (Modes.settle_ms > 0) usleep((useconds_t)Modes.settle_ms * 1000U);
    return Modes.freq;
}

static void modesInitPLUTOSDR(void) {
    int device_count;

    fprintf(stderr, "* Acquiring IIO context\n");
    Modes.ctx = iio_create_default_context();
    if (Modes.ctx == NULL) {
        /* Useful when this same binary is tested from a host PC. */
        Modes.ctx = iio_create_network_context("pluto.local");
    }
    if (Modes.ctx == NULL) {
        fprintf(stderr, "No IIO context. Run locally on the Pluto+ or check IIOD/network access.\n");
        exit(1);
    }

    device_count = (int)iio_context_get_devices_count(Modes.ctx);
    if (!device_count) {
        fprintf(stderr, "No supported IIO devices found.\n");
        exit(1);
    }
    fprintf(stderr, "Found %d IIO device(s).\n", device_count);

    fprintf(stderr, "* Acquiring AD9361 streaming device cf-ad9361-lpc\n");
    Modes.dev = iio_context_find_device(Modes.ctx, "cf-ad9361-lpc");
    if (Modes.dev == NULL) {
        fprintf(stderr, "Could not find cf-ad9361-lpc.\n");
        exit(1);
    }

    fprintf(stderr, "* Acquiring AD9361 phy device ad9361-phy\n");
    Modes.phy = iio_context_find_device(Modes.ctx, "ad9361-phy");
    if (Modes.phy == NULL) {
        fprintf(stderr, "Could not find ad9361-phy.\n");
        exit(1);
    }

    Modes.rx0_phy = iio_device_find_channel(Modes.phy, "voltage0", false);
    if (Modes.rx0_phy == NULL) {
        fprintf(stderr, "Could not find RX phy channel voltage0.\n");
        exit(1);
    }

    Modes.lo_chn = iio_device_find_channel(Modes.phy, "altvoltage0", true);
    if (Modes.lo_chn == NULL) {
        fprintf(stderr, "Could not find RX LO channel altvoltage0.\n");
        exit(1);
    }

    attr_write_str(Modes.rx0_phy, "rf_port_select", "A_BALANCED");

    /*
     * Pluto/AD9361 often rejects 2.000 MSPS unless a custom FIR filter is
     * loaded.  Query sampling_frequency_available and clamp before writing to
     * avoid: sampling_frequency: Invalid argument (-22).
     */
    Modes.sample_rate = clamp_to_available_rate(Modes.rx0_phy, Modes.sample_rate);
    attr_write_ll(Modes.rx0_phy, "sampling_frequency", Modes.sample_rate);

    /* Keep RF bandwidth valid for the selected firmware/device. */
    if (Modes.rf_bandwidth <= 0) Modes.rf_bandwidth = Modes.sample_rate;
    Modes.rf_bandwidth = clamp_rf_bandwidth(Modes.rx0_phy, Modes.rf_bandwidth);
    attr_write_ll(Modes.rx0_phy, "rf_bandwidth", Modes.rf_bandwidth);

    /*
     * Do not call ad9361_set_bb_rate() here.
     *
     * The original dump1090 Pluto fork can use libad9361-iio, but many Pluto+
     * images only ship libiio.  Writing the normal IIO attributes directly is
     * enough for this scanner and avoids an undefined-reference link failure
     * when libad9361-iio is missing or not linked.
     */

    if (Modes.gain == MODES_AUTO_GAIN || Modes.enable_agc) {
        attr_write_str(Modes.rx0_phy, "gain_control_mode", "slow_attack");
    } else {
        char gain_str[32];
        if (Modes.gain > MODES_MAX_GAIN) Modes.gain = MODES_MAX_GAIN;
        attr_write_str(Modes.rx0_phy, "gain_control_mode", "manual");
        snprintf(gain_str, sizeof(gain_str), "%.2f", Modes.gain);
        attr_write_str(Modes.rx0_phy, "hardwaregain", gain_str);
    }

    fprintf(stderr, "* Initializing IIO streaming channels voltage0 / voltage1\n");
    Modes.rx0_i = iio_device_find_channel(Modes.dev, "voltage0", false);
    Modes.rx0_q = iio_device_find_channel(Modes.dev, "voltage1", false);
    if (!Modes.rx0_i || !Modes.rx0_q) {
        fprintf(stderr, "Could not find streaming channels voltage0 and voltage1.\n");
        exit(1);
    }
    iio_channel_enable(Modes.rx0_i);
    iio_channel_enable(Modes.rx0_q);

    fprintf(stderr, "* Creating non-cyclic IIO RX buffer\n");
    Modes.rxbuf = iio_device_create_buffer(Modes.dev, (size_t)Modes.fft_size, false);
    if (!Modes.rxbuf) {
        perror("Could not create RX buffer");
        exit(1);
    }
}

static void debugPrintIioSettings(void) {
    char buf[256];
    long long llv;
    if (!Modes.debug_iio) return;

    fprintf(stderr, "\n--- Pluto/IIO settings ---\n");
    if (Modes.lo_chn && iio_channel_attr_read_longlong(Modes.lo_chn, "frequency", &llv) >= 0)
        fprintf(stderr, "rx_lo.frequency=%lld\n", llv);
    if (Modes.rx0_phy && iio_channel_attr_read_longlong(Modes.rx0_phy, "sampling_frequency", &llv) >= 0)
        fprintf(stderr, "voltage0.sampling_frequency=%lld\n", llv);
    if (Modes.rx0_phy && iio_channel_attr_read_longlong(Modes.rx0_phy, "rf_bandwidth", &llv) >= 0)
        fprintf(stderr, "voltage0.rf_bandwidth=%lld\n", llv);
    if (Modes.rx0_phy && attr_read_str(Modes.rx0_phy, "rf_port_select", buf, sizeof(buf)) >= 0)
        fprintf(stderr, "voltage0.rf_port_select=%s\n", trimWhitespace(buf));
    if (Modes.rx0_phy && attr_read_str(Modes.rx0_phy, "gain_control_mode", buf, sizeof(buf)) >= 0)
        fprintf(stderr, "voltage0.gain_control_mode=%s\n", trimWhitespace(buf));
    if (Modes.rx0_phy && attr_read_str(Modes.rx0_phy, "hardwaregain", buf, sizeof(buf)) >= 0)
        fprintf(stderr, "voltage0.hardwaregain=%s\n", trimWhitespace(buf));
    if (Modes.rx0_phy && attr_read_str(Modes.rx0_phy, "sampling_frequency_available", buf, sizeof(buf)) >= 0)
        fprintf(stderr, "voltage0.sampling_frequency_available=%s\n", trimWhitespace(buf));
    if (Modes.rx0_phy && attr_read_str(Modes.rx0_phy, "rf_bandwidth_available", buf, sizeof(buf)) >= 0)
        fprintf(stderr, "voltage0.rf_bandwidth_available=%s\n", trimWhitespace(buf));
    fprintf(stderr, "--------------------------\n\n");
}

static int cmp_air_channel_freq(const void *a, const void *b) {
    const struct air_channel *ca = (const struct air_channel*)a;
    const struct air_channel *cb = (const struct air_channel*)b;
    if (ca->freq < cb->freq) return -1;
    if (ca->freq > cb->freq) return 1;
    return 0;
}

static int addChannel(struct air_channel **list, int *count, int *cap,
                      long long hz, const char *label) {
    if (hz <= 0) return -1;
    if (*count >= *cap) {
        int newcap = (*cap == 0) ? 64 : (*cap * 2);
        struct air_channel *tmp = realloc(*list, (size_t)newcap * sizeof(struct air_channel));
        if (!tmp) return -1;
        *list = tmp;
        *cap = newcap;
    }
    memset(&(*list)[*count], 0, sizeof(struct air_channel));
    (*list)[*count].freq = hz;
    (*list)[*count].peak_score_db = -999.0;
    if (label && *label) {
        snprintf((*list)[*count].label, sizeof((*list)[*count].label), "%s", label);
    }
    (*count)++;
    return 0;
}

static void sortAndDedupeChannels(void) {
    if (!Modes.channels || Modes.channel_count <= 1) return;
    qsort(Modes.channels, (size_t)Modes.channel_count, sizeof(struct air_channel), cmp_air_channel_freq);

    int out = 0;
    for (int i = 0; i < Modes.channel_count; i++) {
        if (out > 0 && Modes.channels[i].freq == Modes.channels[out - 1].freq) {
            if (Modes.channels[out - 1].label[0] == '\0' && Modes.channels[i].label[0] != '\0') {
                char tmp_label[sizeof(Modes.channels[out - 1].label)];
                snprintf(tmp_label, sizeof(tmp_label), "%s", Modes.channels[i].label);
                memcpy(Modes.channels[out - 1].label, tmp_label, sizeof(tmp_label));
            }
            continue;
        }
        if (out != i) Modes.channels[out] = Modes.channels[i];
        out++;
    }
    Modes.channel_count = out;
}

static long long parseFrequencyTokenHz(const char *tok, int *ok) {
    char *end = NULL;
    double v;
    *ok = 0;
    errno = 0;
    v = strtod(tok, &end);
    if (errno || end == tok) return 0;

    while (end && *end && isspace((unsigned char)*end)) end++;
    if (end && *end) {
        if (!strcasecmp(end, "m") || !strcasecmp(end, "mhz")) {
            v *= 1000000.0;
        } else if (!strcasecmp(end, "k") || !strcasecmp(end, "khz")) {
            v *= 1000.0;
        } else if (!strcasecmp(end, "h") || !strcasecmp(end, "hz")) {
            /* already Hz */
        } else {
            return 0;
        }
    } else {
        /* Bare values below 1,000,000 are assumed to be MHz, e.g. 118.700. */
        if (strchr(tok, '.') || v < 1000000.0) v *= 1000000.0;
    }

    if (v <= 0.0 || v > 10000000000.0) return 0;
    *ok = 1;
    return (long long)llround(v);
}

static int loadFrequencyFile(const char *path) {
    FILE *fp = fopen(path, "r");
    char line[512];
    int count = 0, cap = 0, lineno = 0;
    long long minf = 0, maxf = 0;

    if (!fp) {
        fprintf(stderr, "Could not open frequency file '%s': %s\n", path, strerror(errno));
        return -1;
    }

    while (fgets(line, sizeof(line), fp)) {
        char *hash, *p, *tok, *label;
        int ok = 0;
        long long hz;
        lineno++;

        hash = strchr(line, '#');
        if (hash) *hash = '\0';
        p = trimWhitespace(line);
        if (*p == '\0') continue;

        tok = p;
        while (*p && !isspace((unsigned char)*p) && *p != ',') p++;
        if (*p) *p++ = '\0';
        label = trimWhitespace(p);
        if (*label == ',') label = trimWhitespace(label + 1);

        hz = parseFrequencyTokenHz(tok, &ok);
        if (!ok) {
            fprintf(stderr, "Ignoring invalid frequency file line %d: %s\n", lineno, tok);
            continue;
        }

        if (addChannel(&Modes.channels, &count, &cap, hz, label) < 0) {
            fclose(fp);
            return -1;
        }
        if (minf == 0 || hz < minf) minf = hz;
        if (maxf == 0 || hz > maxf) maxf = hz;
    }
    fclose(fp);

    Modes.channel_count = count;
    if (Modes.channel_count <= 0) {
        fprintf(stderr, "Frequency file '%s' did not contain any usable channels.\n", path);
        return -1;
    }

    sortAndDedupeChannels();

    if (!Modes.scan_start_set) Modes.scan_start = minf;
    if (!Modes.scan_end_set) Modes.scan_end = maxf;

    fprintf(stderr, "Loaded %d watch-list channel(s) from %s.\n", Modes.channel_count, path);
    return 0;
}

static int buildChannelList(void) {
    long long f;
    int n = 0, cap = 0;

    if (Modes.freq_file) return loadFrequencyFile(Modes.freq_file);
    if (Modes.channel_step <= 0 || Modes.scan_end < Modes.scan_start) return -1;

    for (f = Modes.scan_start; f <= Modes.scan_end; f += Modes.channel_step) n++;
    Modes.channels = calloc((size_t)n, sizeof(struct air_channel));
    if (!Modes.channels) return -1;

    n = 0;
    cap = n;
    MODES_NOTUSED(cap);
    for (f = Modes.scan_start; f <= Modes.scan_end; f += Modes.channel_step) {
        Modes.channels[n].freq = f;
        Modes.channels[n].last_score_db = 0.0;
        Modes.channels[n].peak_score_db = -999.0;
        n++;
    }
    Modes.channel_count = n;
    return 0;
}

static void fft_forward(cpxf *a, int n) {
    int i, j;
    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            cpxf t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        float ang = (float)(-2.0 * M_PI / (double)len);
        cpxf wlen = { cosf(ang), sinf(ang) };
        for (i = 0; i < n; i += len) {
            cpxf w = { 1.0f, 0.0f };
            for (j = 0; j < len / 2; j++) {
                cpxf u = a[i + j];
                cpxf v = {
                    a[i + j + len / 2].re * w.re - a[i + j + len / 2].im * w.im,
                    a[i + j + len / 2].re * w.im + a[i + j + len / 2].im * w.re
                };
                a[i + j].re = u.re + v.re;
                a[i + j].im = u.im + v.im;
                a[i + j + len / 2].re = u.re - v.re;
                a[i + j + len / 2].im = u.im - v.im;

                float wr = w.re * wlen.re - w.im * wlen.im;
                float wi = w.re * wlen.im + w.im * wlen.re;
                w.re = wr;
                w.im = wi;
            }
        }
    }
}

static int readSamplesIntoFFT(cpxf *fftbuf, int n) {
    ssize_t ret;
    void *p_dat, *p_end;
    ptrdiff_t p_inc;
    int count = 0;
    double mean_i = 0.0, mean_q = 0.0;

    ret = iio_buffer_refill(Modes.rxbuf);
    if (ret < 0) return (int)ret;

    p_inc = iio_buffer_step(Modes.rxbuf);
    p_end = iio_buffer_end(Modes.rxbuf);

    for (p_dat = iio_buffer_first(Modes.rxbuf, Modes.rx0_i);
         p_dat < p_end && count < n;
         p_dat += p_inc) {
        const int16_t si = ((int16_t*)p_dat)[0];
        const int16_t sq = ((int16_t*)p_dat)[1];
        fftbuf[count].re = (float)si / 32768.0f;
        fftbuf[count].im = (float)sq / 32768.0f;
        mean_i += fftbuf[count].re;
        mean_q += fftbuf[count].im;
        count++;
    }

    if (count <= 0) return -EIO;

    mean_i /= count;
    mean_q /= count;
    for (int i = 0; i < count; i++) {
        double w = 0.5 - 0.5 * cos((2.0 * M_PI * i) / (double)(count - 1));
        fftbuf[i].re = (float)((fftbuf[i].re - mean_i) * w);
        fftbuf[i].im = (float)((fftbuf[i].im - mean_q) * w);
    }
    for (int i = count; i < n; i++) {
        fftbuf[i].re = 0.0f;
        fftbuf[i].im = 0.0f;
    }

    /* Fill the template raw buffer for optional envelope debugging. */
    if (Modes.data && Modes.data_len >= (uint32_t)(2 * count)) {
        for (int i = 0; i < count; i++) {
            int ii = (int)lrintf(fftbuf[i].re * 64.0f) + 127;
            int qq = (int)lrintf(fftbuf[i].im * 64.0f) + 127;
            if (ii < 0) ii = 0;
            if (ii > 255) ii = 255;
            if (qq < 0) qq = 0;
            if (qq > 255) qq = 255;
            Modes.data[2*i] = (unsigned char)ii;
            Modes.data[2*i + 1] = (unsigned char)qq;
        }
        computeMagnitudeVector();
    }

    return count;
}

static int capturePowerSpectrum(float *power, int n, int avg_count) {
    cpxf *fftbuf = calloc((size_t)n, sizeof(cpxf));
    if (!fftbuf) return -ENOMEM;
    memset(power, 0, (size_t)n * sizeof(float));

    for (int a = 0; a < avg_count; a++) {
        int got = readSamplesIntoFFT(fftbuf, n);
        if (got < 0) {
            free(fftbuf);
            return got;
        }
        fft_forward(fftbuf, n);
        for (int i = 0; i < n; i++) {
            power[i] += fftbuf[i].re * fftbuf[i].re + fftbuf[i].im * fftbuf[i].im;
        }
    }

    for (int i = 0; i < n; i++) power[i] /= (float)avg_count;
    free(fftbuf);
    return 0;
}

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static float percentile_copy(const float *x, int n, double pct) {
    float *tmp = malloc((size_t)n * sizeof(float));
    int idx;
    if (!tmp) return 1e-20f;
    memcpy(tmp, x, (size_t)n * sizeof(float));
    qsort(tmp, (size_t)n, sizeof(float), cmp_float);
    if (pct < 0.0) pct = 0.0;
    if (pct > 1.0) pct = 1.0;
    idx = (int)floor(pct * (double)(n - 1));
    float v = tmp[idx];
    free(tmp);
    if (v <= 1e-20f) v = 1e-20f;
    return v;
}

static int offsetToBin(double offset_hz, int n, double sample_rate) {
    long k = lround(offset_hz * (double)n / sample_rate);
    k %= n;
    if (k < 0) k += n;
    return (int)k;
}

static double channelScoreDb(const float *power, int n, long long center_hz, long long chan_hz,
                             float noise_floor) {
    double offset = (double)(chan_hz - center_hz);
    int center_bin = offsetToBin(offset, n, (double)Modes.sample_rate);
    int half_bins = (int)ceil((Modes.detect_bw_hz / 2.0) * (double)n / (double)Modes.sample_rate);
    float peak = 1e-20f;

    if (half_bins < 1) half_bins = 1;
    for (int d = -half_bins; d <= half_bins; d++) {
        int b = center_bin + d;
        while (b < 0) b += n;
        while (b >= n) b -= n;
        if (power[b] > peak) peak = power[b];
    }
    if (noise_floor <= 1e-20f) noise_floor = 1e-20f;
    return 10.0 * log10((double)peak / (double)noise_floor);
}

static long long channelCurrentActiveMs(const struct air_channel *ch, long long now_ms) {
    long long total = ch->total_active_ms;
    if (ch->active && ch->active_start_ms > 0 && now_ms >= ch->active_start_ms) {
        total += now_ms - ch->active_start_ms;
    }
    return total;
}

static void writeCsvEvent(const char *event, const struct air_channel *ch,
                          double score_db, long long now_ms,
                          long long duration_ms, double peak_db) {
    char tbuf[64];
    if (!Modes.csv_log) return;
    formatTimeMs(now_ms, tbuf, sizeof(tbuf));
    csvPrintEscaped(Modes.csv_log, tbuf);
    fprintf(Modes.csv_log, ",");
    csvPrintEscaped(Modes.csv_log, event);
    fprintf(Modes.csv_log, ",%lld,%.6f,", ch->freq, (double)ch->freq / 1000000.0);
    csvPrintEscaped(Modes.csv_log, ch->label);
    fprintf(Modes.csv_log, ",%.2f,%.2f,%lld,%lld,%lu,%lu\n",
            score_db,
            peak_db,
            duration_ms,
            channelCurrentActiveMs(ch, now_ms),
            ch->hit_count,
            ch->activation_count);
    fflush(Modes.csv_log);
}

static void chirpMakeName(const struct air_channel *ch, char *out, size_t len) {
    char tmp[128];
    size_t j = 0;

    if (!out || len == 0) return;
    out[0] = '\0';

    if (ch->label[0]) {
        snprintf(tmp, sizeof(tmp), "%s", ch->label);
    } else {
        snprintf(tmp, sizeof(tmp), "AIR%06lld", ch->freq / 1000);
    }

    /* Keep names conservative for handheld/mobile radios after CHIRP import:
     * uppercase A-Z, digits, dash, underscore.  Generic CSV itself accepts
     * much longer names, but many real radios truncate heavily.
     */
    for (size_t i = 0; tmp[i] && j + 1 < len && j < 16; i++) {
        unsigned char c = (unsigned char)tmp[i];
        if (isalnum(c)) {
            out[j++] = (char)toupper(c);
        } else if ((c == '-' || c == '_' || c == '/') && j > 0) {
            out[j++] = (c == '/') ? '-' : (char)c;
        } else if (isspace(c) && j > 0 && out[j - 1] != '_') {
            out[j++] = '_';
        }
    }
    while (j > 0 && (out[j - 1] == '_' || out[j - 1] == '-')) j--;
    out[j] = '\0';

    if (out[0] == '\0') {
        snprintf(out, len, "AIR%06lld", ch->freq / 1000);
    }
}

static void writeChirpMemory(struct air_channel *ch, double score_db,
                             long long now_ms, double peak_db) {
    char name[32];
    char tbuf[64];
    char comment[256];
    double freq_mhz;
    double step_khz;

    if (!Modes.chirp_csv || !ch || ch->chirp_exported) return;

    chirpMakeName(ch, name, sizeof(name));
    formatTimeMs(now_ms, tbuf, sizeof(tbuf));

    freq_mhz = (double)ch->freq / 1000000.0;
    step_khz = (double)Modes.channel_step / 1000.0;
    if (step_khz <= 0.0) step_khz = 25.0;

    if (ch->label[0]) {
        snprintf(comment, sizeof(comment),
                 "Airband detected %s; label=%s; score=%.1f dB; peak=%.1f dB",
                 tbuf, ch->label, score_db, peak_db);
    } else {
        snprintf(comment, sizeof(comment),
                 "Airband detected %s; score=%.1f dB; peak=%.1f dB",
                 tbuf, score_db, peak_db);
    }

    /* CHIRP Generic CSV columns:
     * Location,Name,Frequency,Duplex,Offset,Tone,rToneFreq,cToneFreq,
     * DtcsCode,DtcsPolarity,Mode,TStep,Skip,Comment,URCALL,RPT1CALL,
     * RPT2CALL,DVCODE
     *
     * Duplex=off makes the channel receive-only where the target radio
     * supports it, which is safer for civil airband monitoring.
     */
    fprintf(Modes.chirp_csv, "%d,", Modes.chirp_next_location++);
    csvPrintEscaped(Modes.chirp_csv, name);
    fprintf(Modes.chirp_csv, ",%.6f,off,0.000000,,88.5,88.5,023,NN,AM,%.2f,,",
            freq_mhz, step_khz);
    csvPrintEscaped(Modes.chirp_csv, comment);
    fprintf(Modes.chirp_csv, ",,,,\n");
    fflush(Modes.chirp_csv);

    ch->chirp_exported = 1;
    Modes.chirp_export_count++;

    if (Modes.verbose || Modes.debug_iio) {
        fprintf(stderr, "CHIRP export: location=%d frequency=%.6f MHz name=%s\n",
                Modes.chirp_next_location - 1, freq_mhz, name);
    }
}

static void emitEvent(const char *event, const struct air_channel *ch,
                      double score_db, long long now_ms,
                      long long duration_ms, double peak_db) {
    char tbuf[64];
    char dbuf[32];
    formatTimeMs(now_ms, tbuf, sizeof(tbuf));
    formatDurationMs(duration_ms, dbuf, sizeof(dbuf));

    if (Modes.json_output) {
        printf("{\"time\":");
        jsonPrintEscaped(stdout, tbuf);
        printf(",\"event\":");
        jsonPrintEscaped(stdout, event);
        printf(",\"freq_hz\":%lld,\"freq_mhz\":%.6f", ch->freq, (double)ch->freq / 1000000.0);
        if (ch->label[0]) {
            printf(",\"label\":");
            jsonPrintEscaped(stdout, ch->label);
        }
        printf(",\"score_db\":%.2f,\"peak_db\":%.2f", score_db, peak_db);
        if (!strcmp(event, "ENDED")) {
            printf(",\"duration_ms\":%lld,\"duration_s\":%.3f,\"total_active_ms\":%lld",
                   duration_ms, (double)duration_ms / 1000.0, channelCurrentActiveMs(ch, now_ms));
        }
        printf(",\"hit_count\":%lu,\"activation_count\":%lu}\n",
               ch->hit_count, ch->activation_count);
    } else if (!strcmp(event, "ACTIVE")) {
        if (ch->label[0]) {
            printf("%s  ACTIVE  %.3f MHz  score=%.1f dB  %s\n",
                   tbuf, (double)ch->freq / 1000000.0, score_db, ch->label);
        } else {
            printf("%s  ACTIVE  %.3f MHz  score=%.1f dB\n",
                   tbuf, (double)ch->freq / 1000000.0, score_db);
        }
    } else {
        if (ch->label[0]) {
            printf("%s  ENDED   %.3f MHz  duration=%s  peak=%.1f dB  total=%llds  %s\n",
                   tbuf, (double)ch->freq / 1000000.0, dbuf, peak_db,
                   channelCurrentActiveMs(ch, now_ms) / 1000, ch->label);
        } else {
            printf("%s  ENDED   %.3f MHz  duration=%s  peak=%.1f dB  total=%llds\n",
                   tbuf, (double)ch->freq / 1000000.0, dbuf, peak_db,
                   channelCurrentActiveMs(ch, now_ms) / 1000);
        }
    }
    fflush(stdout);
    writeCsvEvent(event, ch, score_db, now_ms, duration_ms, peak_db);
}

static void updateChannelState(struct air_channel *ch, double score_db, long long now_ms) {
    int hit = (score_db >= Modes.threshold_db);
    ch->last_score_db = score_db;
    ch->last_tested_ms = now_ms;

    if (Modes.debug_scores) {
        fprintf(stderr, "SCORE  %.6f MHz  score=%6.2f dB  threshold=%5.2f  state=%s%s%s\n",
                (double)ch->freq / 1000000.0,
                score_db,
                Modes.threshold_db,
                ch->active ? "ACTIVE" : "inactive",
                ch->label[0] ? "  " : "",
                ch->label[0] ? ch->label : "");
    }

    if (hit) {
        ch->hit_count++;
        ch->last_seen_ms = now_ms;
        ch->hit_streak++;
        ch->miss_streak = 0;
        if (ch->hit_streak == 1) ch->first_hit_ms = now_ms;
        if (score_db > ch->peak_score_db) ch->peak_score_db = score_db;

        if (ch->active) {
            /* Continue current transmission. */
            return;
        }

        if (ch->hit_streak >= Modes.active_hits) {
            ch->active = 1;
            ch->activation_count++;
            ch->active_start_ms = ch->first_hit_ms ? ch->first_hit_ms : now_ms;
            writeChirpMemory(ch, score_db, now_ms, ch->peak_score_db);
            emitEvent("ACTIVE", ch, score_db, now_ms, 0, ch->peak_score_db);
        }
    } else {
        ch->miss_count++;
        ch->miss_streak++;
        ch->hit_streak = 0;
        ch->first_hit_ms = 0;

        if (ch->active && ch->miss_streak >= Modes.inactive_misses) {
            long long duration_ms = now_ms - ch->active_start_ms;
            if (duration_ms < 0) duration_ms = 0;
            ch->total_active_ms += duration_ms;
            emitEvent("ENDED", ch, score_db, now_ms, duration_ms, ch->peak_score_db);
            ch->active = 0;
            ch->active_start_ms = 0;
        }
    }
}

static void closeActiveChannels(void) {
    long long now_ms = mstime();
    for (int i = 0; i < Modes.channel_count; i++) {
        struct air_channel *ch = &Modes.channels[i];
        if (ch->active) {
            long long duration_ms = now_ms - ch->active_start_ms;
            if (duration_ms < 0) duration_ms = 0;
            ch->total_active_ms += duration_ms;
            emitEvent("ENDED", ch, ch->last_score_db, now_ms, duration_ms, ch->peak_score_db);
            ch->active = 0;
            ch->active_start_ms = 0;
        }
    }
}

static int openCsvLog(void) {
    if (!Modes.csv_log_path) return 0;
    Modes.csv_log = fopen(Modes.csv_log_path, "a");
    if (!Modes.csv_log) {
        fprintf(stderr, "Could not open CSV log '%s': %s\n", Modes.csv_log_path, strerror(errno));
        return -1;
    }
    if (ftell(Modes.csv_log) == 0) {
        fprintf(Modes.csv_log,
                "time,event,freq_hz,freq_mhz,label,score_db,peak_db,duration_ms,total_active_ms,hit_count,activation_count\n");
        fflush(Modes.csv_log);
    }
    return 0;
}

static int openChirpCsv(void) {
    if (!Modes.chirp_csv_path) return 0;

    Modes.chirp_csv = fopen(Modes.chirp_csv_path, "w");
    if (!Modes.chirp_csv) {
        fprintf(stderr, "Could not open CHIRP CSV '%s': %s\n",
                Modes.chirp_csv_path, strerror(errno));
        return -1;
    }

    if (Modes.chirp_start_location < 0) Modes.chirp_start_location = 0;
    Modes.chirp_next_location = Modes.chirp_start_location;

    fprintf(Modes.chirp_csv,
            "Location,Name,Frequency,Duplex,Offset,Tone,rToneFreq,cToneFreq,"
            "DtcsCode,DtcsPolarity,Mode,TStep,Skip,Comment,"
            "URCALL,RPT1CALL,RPT2CALL,DVCODE\n");
    fflush(Modes.chirp_csv);

    if (Modes.verbose || Modes.debug_iio) {
        fprintf(stderr, "Writing CHIRP Generic CSV memories to %s starting at location %d\n",
                Modes.chirp_csv_path, Modes.chirp_start_location);
    }
    return 0;
}

static void discardBuffers(int count) {
    for (int i = 0; i < count; i++) {
        ssize_t ret = iio_buffer_refill(Modes.rxbuf);
        if (ret < 0) return;
    }
}

static void computeUsableWindow(long long *usable_span_out, long long *half_usable_out) {
    double usable_bw_d = (double)Modes.sample_rate * Modes.usable_fraction;
    long long usable_span = (long long)llround(usable_bw_d);
    long long band_span = Modes.scan_end - Modes.scan_start;

    if (usable_span <= 0) usable_span = Modes.sample_rate / 2;
    if (usable_span <= 0) usable_span = Modes.channel_step;

    /* If the requested span is smaller than one normal sample window, keep the
     * larger sample-derived window.  This is important for one-frequency watch
     * lists, where scan_start == scan_end.
     */
    if (band_span > Modes.channel_step && usable_span > band_span) usable_span = band_span;
    if (usable_span < Modes.channel_step) usable_span = Modes.channel_step;

    *usable_span_out = usable_span;
    *half_usable_out = usable_span / 2;
    if (*half_usable_out <= 0) *half_usable_out = Modes.channel_step / 2;
    if (*half_usable_out <= 0) *half_usable_out = 1;
}

static int scanWindow(long long center, long long half_usable, float *power, const char *tag) {
    long long tuned_center;
    long long low = center - half_usable;
    long long high = center + half_usable;
    long long actual_low;
    long long actual_high;
    int channels_in_window = 0;
    int channels_tested = 0;
    double max_score = -999.0;
    long long max_score_freq = 0;
    int ret;
    char tagbuf[64] = "";

    if (tag && *tag) snprintf(tagbuf, sizeof(tagbuf), "[%s]", tag);

    if (low < Modes.scan_start) low = Modes.scan_start;
    if (high > Modes.scan_end) high = Modes.scan_end;

    if (Modes.verbose) {
        fprintf(stderr,
                "TUNING%s center=%.6f MHz  window=%.6f-%.6f MHz\n",
                tagbuf,
                (double)center / 1000000.0,
                (double)low / 1000000.0,
                (double)high / 1000000.0);
    }

    tuned_center = setCenterFrequency(center);
    discardBuffers(2);

    actual_low = tuned_center - half_usable;
    actual_high = tuned_center + half_usable;
    if (actual_low < Modes.scan_start) actual_low = Modes.scan_start;
    if (actual_high > Modes.scan_end) actual_high = Modes.scan_end;

    ret = capturePowerSpectrum(power, Modes.fft_size, Modes.fft_avg);
    if (ret < 0) return ret;

    float floor_power = percentile_copy(power, Modes.fft_size, 0.35);
    long long now_ms = mstime();

    for (int c = 0; c < Modes.channel_count; c++) {
        long long f = Modes.channels[c].freq;
        if (f < actual_low || f > actual_high) continue;
        channels_in_window++;

        /* Avoid the residual DC spike at exactly zero IF.  Priority windows are
         * deliberately offset so the target channel is not skipped here.
         */
        if (llabs(f - tuned_center) < (Modes.channel_step / 2)) continue;
        channels_tested++;

        double score = channelScoreDb(power, Modes.fft_size, tuned_center, f, floor_power);
        if (score > max_score) {
            max_score = score;
            max_score_freq = f;
        }
        updateChannelState(&Modes.channels[c], score, now_ms);
    }

    if (Modes.verbose) {
        if (channels_tested > 0) {
            fprintf(stderr,
                    "TUNED%s center=%.6f MHz  actual-window=%.6f-%.6f MHz  channels=%d tested=%d  max=%.1f dB @ %.3f MHz\n",
                    tagbuf,
                    (double)tuned_center / 1000000.0,
                    (double)actual_low / 1000000.0,
                    (double)actual_high / 1000000.0,
                    channels_in_window,
                    channels_tested,
                    max_score,
                    (double)max_score_freq / 1000000.0);
        } else {
            fprintf(stderr,
                    "TUNED%s center=%.6f MHz  actual-window=%.6f-%.6f MHz  channels=%d tested=%d\n",
                    tagbuf,
                    (double)tuned_center / 1000000.0,
                    (double)actual_low / 1000000.0,
                    (double)actual_high / 1000000.0,
                    channels_in_window,
                    channels_tested);
        }
        fflush(stderr);
    }

    return channels_tested;
}

static int cmp_summary_ptrs(const void *a, const void *b) {
    const struct air_channel *ca = *(const struct air_channel * const *)a;
    const struct air_channel *cb = *(const struct air_channel * const *)b;
    if (ca->active != cb->active) return cb->active - ca->active;
    if (ca->hit_count < cb->hit_count) return 1;
    if (ca->hit_count > cb->hit_count) return -1;
    if (ca->peak_score_db < cb->peak_score_db) return 1;
    if (ca->peak_score_db > cb->peak_score_db) return -1;
    if (ca->freq < cb->freq) return -1;
    if (ca->freq > cb->freq) return 1;
    return 0;
}

static void printSummary(long long now_ms, int force) {
    struct air_channel **list;
    int used = 0;
    int limit;
    char tbuf[64];

    if (!force && Modes.summary_every_ms <= 0) return;
    if (!force && Modes.last_summary_ms && now_ms - Modes.last_summary_ms < Modes.summary_every_ms) return;
    Modes.last_summary_ms = now_ms;

    list = calloc((size_t)Modes.channel_count, sizeof(*list));
    if (!list) return;
    for (int i = 0; i < Modes.channel_count; i++) {
        if (Modes.channels[i].hit_count > 0 || Modes.channels[i].active) {
            list[used++] = &Modes.channels[i];
        }
    }
    qsort(list, (size_t)used, sizeof(*list), cmp_summary_ptrs);

    formatTimeMs(now_ms, tbuf, sizeof(tbuf));
    fprintf(stderr, "\n--- Airband activity summary @ %s ---\n", tbuf);
    if (used == 0) {
        fprintf(stderr, "No activity detected yet.\n\n");
        free(list);
        return;
    }

    limit = Modes.summary_top;
    if (limit <= 0 || limit > used) limit = used;
    fprintf(stderr, "%-11s %-6s %-7s %-9s %-19s %-11s %s\n",
            "Frequency", "State", "Hits", "Peak", "Last seen", "Active", "Label");
    for (int i = 0; i < limit; i++) {
        struct air_channel *ch = list[i];
        char lastbuf[64] = "-";
        char durbuf[32];
        if (ch->last_seen_ms > 0) formatTimeMs(ch->last_seen_ms, lastbuf, sizeof(lastbuf));
        formatDurationMs(channelCurrentActiveMs(ch, now_ms), durbuf, sizeof(durbuf));
        fprintf(stderr, "%-11.3f %-6s %-7lu %-9.1f %-19s %-11s %s\n",
                (double)ch->freq / 1000000.0,
                ch->active ? "ACTIVE" : "idle",
                ch->hit_count,
                ch->peak_score_db,
                lastbuf,
                durbuf,
                ch->label);
    }
    fprintf(stderr, "\n");
    free(list);
}

static int scanOnePass(void) {
    int n = Modes.fft_size;
    float *power = malloc((size_t)n * sizeof(float));
    if (!power) return -ENOMEM;

    long long usable_span;
    long long half_usable;
    long long band_span = Modes.scan_end - Modes.scan_start;
    int windows = 0;

    computeUsableWindow(&usable_span, &half_usable);

    if (band_span <= usable_span) {
        long long center = (Modes.scan_start + Modes.scan_end) / 2;
        long long margin = usable_span - band_span;
        if (margin > Modes.channel_step && Modes.channel_step > 1) {
            /* Shift by a half-channel so a one-frequency watch list is not
             * placed directly on the receiver DC bin.
             */
            center += Modes.channel_step / 2;
        }
        int ret = scanWindow(center, half_usable, power, "full");
        free(power);
        if (ret < 0) return ret;
        return 1;
    }

    long long base_center = Modes.scan_start + half_usable;
    while (!Modes.exit) {
        long long center = base_center;
        int ret;

        if (center + half_usable > Modes.scan_end) center = Modes.scan_end - half_usable;
        if (center - half_usable < Modes.scan_start) center = Modes.scan_start + half_usable;
        if (center < Modes.scan_start || center > Modes.scan_end) center = (Modes.scan_start + Modes.scan_end) / 2;

        ret = scanWindow(center, half_usable, power, "band");
        if (ret < 0) {
            free(power);
            return ret;
        }

        windows++;
        if (center + half_usable >= Modes.scan_end) break;

        base_center += usable_span;
        if (base_center > Modes.scan_end + half_usable) break;
    }

    free(power);
    return windows;
}

static int cmp_priority_ptrs(const void *a, const void *b) {
    const struct air_channel *ca = *(const struct air_channel * const *)a;
    const struct air_channel *cb = *(const struct air_channel * const *)b;
    if (ca->active != cb->active) return cb->active - ca->active;
    if (ca->last_seen_ms < cb->last_seen_ms) return 1;
    if (ca->last_seen_ms > cb->last_seen_ms) return -1;
    if (ca->peak_score_db < cb->peak_score_db) return 1;
    if (ca->peak_score_db > cb->peak_score_db) return -1;
    return 0;
}

static long long priorityCenterForFrequency(long long f, long long half_usable) {
    long long offset = half_usable / 2;
    long long center;
    if (offset < Modes.channel_step) offset = Modes.channel_step;
    center = f + offset;

    if (Modes.scan_end > Modes.scan_start) {
        if (center + half_usable > Modes.scan_end) center = f - offset;
        if (center - half_usable < Modes.scan_start) center = f + offset;
    }
    if (llabs(center - f) < (Modes.channel_step / 2)) center = f + Modes.channel_step;
    return center;
}

static int scanPriorityChannels(void) {
    long long now_ms = mstime();
    long long usable_span, half_usable;
    struct air_channel **pri;
    int count = 0;
    int limit;
    float *power;
    int scanned = 0;

    if (!Modes.priority_mode) return 0;
    computeUsableWindow(&usable_span, &half_usable);
    MODES_NOTUSED(usable_span);

    pri = calloc((size_t)Modes.channel_count, sizeof(*pri));
    if (!pri) return -ENOMEM;
    for (int i = 0; i < Modes.channel_count; i++) {
        struct air_channel *ch = &Modes.channels[i];
        if (ch->active || (ch->last_seen_ms > 0 && now_ms - ch->last_seen_ms <= Modes.priority_recent_ms)) {
            pri[count++] = ch;
        }
    }
    if (count == 0) {
        free(pri);
        return 0;
    }

    qsort(pri, (size_t)count, sizeof(*pri), cmp_priority_ptrs);
    limit = Modes.priority_max_channels;
    if (limit <= 0 || limit > count) limit = count;

    power = malloc((size_t)Modes.fft_size * sizeof(float));
    if (!power) {
        free(pri);
        return -ENOMEM;
    }

    for (int i = 0; i < limit && !Modes.exit; i++) {
        long long center = priorityCenterForFrequency(pri[i]->freq, half_usable);
        int ret = scanWindow(center, half_usable, power, "priority");
        if (ret < 0) {
            free(power);
            free(pri);
            return ret;
        }
        scanned++;
    }

    free(power);
    free(pri);
    return scanned;
}

static void showHelp(void) {
    printf(
        "Pluto+ airband activity scanner\n"
        "\n"
        "Basic scan options:\n"
        "  --start <Hz>             Start frequency, default 118000000\n"
        "  --end <Hz>               End frequency, default 136975000\n"
        "  --step <Hz>              Channel step, default 25000\n"
        "  --freq-file <file>       Watch-list file: frequency [label], one per line\n"
        "  --rate <Hz>              Sample rate, default 2400000\n"
        "  --bandwidth <Hz>         RF bandwidth, default = sample rate\n"
        "  --threshold <dB>         Detection threshold over noise, default 16\n"
        "  --detect-bw <Hz>         Detection bandwidth around channel, default 12000\n"
        "  --fft-size <N>           FFT size, power of 2, default 8192\n"
        "  --fft-avg <N>            FFT averages per tuning window, default 4\n"
        "  --gain <dB|-100>         Manual RX gain, or -100 for AGC; default -100\n"
        "  --settle-ms <N>          Retune settling delay, default 15\n"
        "  --once                   Scan band once and exit\n"
        "\n"
        "Stateful activity detection:\n"
        "  --active-hits <N>        Hits before ACTIVE, default 2\n"
        "  --inactive-misses <N>    Misses before ENDED, default 3\n"
        "  --holdoff-ms <N>         Accepted for v4 compatibility; hysteresis now controls repeats\n"
        "\n"
        "Summary and logging:\n"
        "  --summary-every <sec>    Print top activity table every N seconds\n"
        "  --summary-top <N>        Rows in summary table, default 10\n"
        "  --log <file.csv>         Append ACTIVE/ENDED events to CSV\n"
        "  --json                   Print ACTIVE/ENDED events as JSON lines to stdout\n"
        "  --chirp <file.csv>       Write first ACTIVE per frequency as CHIRP Generic CSV\n"
        "  --chirp-start <N>        First CHIRP memory location, default 1\n"
        "\n"
        "Priority scan mode:\n"
        "  --priority               Revisit active/recently-active channels after each band pass\n"
        "  --priority-recent-ms <N> Recent window for priority channels, default 30000\n"
        "  --priority-max <N>       Max priority channels per revisit cycle, default 16\n"
        "\n"
        "Verbose/debug output:\n"
        "  --verbose                Show tuning windows and per-window max score\n"
        "  --debug-scores           Show every tested channel and score\n"
        "  --debug-iio              Show Pluto/libiio settings after initialization\n"
        "  --help                   Show this help\n"
        "\n"
        "Clean shutdown:\n"
        "  Press Ctrl-C, or send SIGTERM/SIGHUP. The scanner exits its loop, closes\n"
        "  active channel events, destroys the IIO buffer, disables RX channels, and\n"
        "  releases the IIO context.\n"
        "\n"
        "Frequency file examples:\n"
        "  118.700  KPHX Tower\n"
        "  120700000 KPHX Approach\n"
        "  121.500  Emergency\n"
        "\n"
        "Example:\n"
        "  ./airband_scan --freq-file phoenix_airband.txt --priority --summary-every 60 --log airband.csv\n"
        "  ./airband_scan --freq-file phoenix_airband.txt --chirp airband_chirp.csv\n"
    );
}

static long long parse_ll(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    long long v = strtoll(s, &end, 10);
    if (errno || !end || *end != '\0') {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        exit(1);
    }
    return v;
}

static double parse_double(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno || !end || *end != '\0') {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        exit(1);
    }
    return v;
}

int main(int argc, char **argv) {
    modesInitConfig();

    for (int j = 1; j < argc; j++) {
        int more = j + 1 < argc;
        if (!strcmp(argv[j], "--start") && more) {
            Modes.scan_start = parse_ll(argv[++j], "start");
            Modes.scan_start_set = 1;
        } else if (!strcmp(argv[j], "--end") && more) {
            Modes.scan_end = parse_ll(argv[++j], "end");
            Modes.scan_end_set = 1;
        } else if (!strcmp(argv[j], "--step") && more) {
            Modes.channel_step = parse_ll(argv[++j], "step");
        } else if (!strcmp(argv[j], "--freq-file") && more) {
            Modes.freq_file = argv[++j];
        } else if (!strcmp(argv[j], "--rate") && more) {
            Modes.sample_rate = parse_ll(argv[++j], "rate");
        } else if (!strcmp(argv[j], "--bandwidth") && more) {
            Modes.rf_bandwidth = parse_ll(argv[++j], "bandwidth");
        } else if (!strcmp(argv[j], "--threshold") && more) {
            Modes.threshold_db = parse_double(argv[++j], "threshold");
        } else if (!strcmp(argv[j], "--detect-bw") && more) {
            Modes.detect_bw_hz = parse_double(argv[++j], "detect-bw");
        } else if (!strcmp(argv[j], "--fft-size") && more) {
            Modes.fft_size = (int)parse_ll(argv[++j], "fft-size");
        } else if (!strcmp(argv[j], "--fft-avg") && more) {
            Modes.fft_avg = (int)parse_ll(argv[++j], "fft-avg");
        } else if (!strcmp(argv[j], "--gain") && more) {
            Modes.gain = parse_double(argv[++j], "gain");
            Modes.enable_agc = (Modes.gain == MODES_AUTO_GAIN) ? 1 : 0;
        } else if (!strcmp(argv[j], "--settle-ms") && more) {
            Modes.settle_ms = (int)parse_ll(argv[++j], "settle-ms");
        } else if (!strcmp(argv[j], "--holdoff-ms") && more) {
            Modes.report_holdoff_ms = parse_ll(argv[++j], "holdoff-ms");
        } else if (!strcmp(argv[j], "--active-hits") && more) {
            Modes.active_hits = (int)parse_ll(argv[++j], "active-hits");
        } else if (!strcmp(argv[j], "--inactive-misses") && more) {
            Modes.inactive_misses = (int)parse_ll(argv[++j], "inactive-misses");
        } else if (!strcmp(argv[j], "--summary-every") && more) {
            Modes.summary_every_ms = parse_ll(argv[++j], "summary-every") * 1000LL;
        } else if (!strcmp(argv[j], "--summary-top") && more) {
            Modes.summary_top = (int)parse_ll(argv[++j], "summary-top");
        } else if (!strcmp(argv[j], "--log") && more) {
            Modes.csv_log_path = argv[++j];
        } else if (!strcmp(argv[j], "--json")) {
            Modes.json_output = 1;
        } else if (!strcmp(argv[j], "--chirp") && more) {
            Modes.chirp_csv_path = argv[++j];
        } else if (!strcmp(argv[j], "--chirp-start") && more) {
            Modes.chirp_start_location = (int)parse_ll(argv[++j], "chirp-start");
        } else if (!strcmp(argv[j], "--priority")) {
            Modes.priority_mode = 1;
        } else if (!strcmp(argv[j], "--priority-recent-ms") && more) {
            Modes.priority_recent_ms = parse_ll(argv[++j], "priority-recent-ms");
        } else if (!strcmp(argv[j], "--priority-max") && more) {
            Modes.priority_max_channels = (int)parse_ll(argv[++j], "priority-max");
        } else if (!strcmp(argv[j], "--once")) {
            Modes.continuous = 0;
        } else if (!strcmp(argv[j], "--verbose")) {
            Modes.verbose = 1;
        } else if (!strcmp(argv[j], "--debug-scores")) {
            Modes.debug_scores = 1;
        } else if (!strcmp(argv[j], "--debug-iio")) {
            Modes.debug_iio = 1;
        } else if (!strcmp(argv[j], "--help")) {
            showHelp();
            return 0;
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n\n", argv[j]);
            showHelp();
            return 1;
        }
    }

    if (Modes.rf_bandwidth <= 0) Modes.rf_bandwidth = Modes.sample_rate;
    if (!is_power_of_two(Modes.fft_size)) {
        fprintf(stderr, "--fft-size must be a power of two.\n");
        return 1;
    }
    if (Modes.fft_avg < 1) Modes.fft_avg = 1;
    if (Modes.active_hits < 1) Modes.active_hits = 1;
    if (Modes.inactive_misses < 1) Modes.inactive_misses = 1;
    if (Modes.summary_top < 1) Modes.summary_top = DEFAULT_SUMMARY_TOP;
    if (Modes.priority_recent_ms < 0) Modes.priority_recent_ms = DEFAULT_PRIORITY_RECENT_MS;
    if (Modes.priority_max_channels < 1) Modes.priority_max_channels = DEFAULT_PRIORITY_MAX;
    if (Modes.usable_fraction <= 0.0 || Modes.usable_fraction > 0.95) Modes.usable_fraction = DEFAULT_USABLE_FRACTION;

    atexit(modesCleanup);

    if (openCsvLog() < 0) return 1;
    if (openChirpCsv() < 0) return 1;

    installSignalHandlers();

    modesInit();
    if (buildChannelList() < 0) {
        fprintf(stderr, "Could not build channel list.\n");
        return 1;
    }

    fprintf(stderr, "Airband scanner: %.3f-%.3f MHz, step %.3f kHz, %d channels\n",
            (double)Modes.scan_start / 1000000.0,
            (double)Modes.scan_end / 1000000.0,
            (double)Modes.channel_step / 1000.0,
            Modes.channel_count);
    fprintf(stderr, "Rate %.3f MSPS, FFT %d x %d, threshold %.1f dB, hysteresis %d/%d\n",
            (double)Modes.sample_rate / 1000000.0,
            Modes.fft_size,
            Modes.fft_avg,
            Modes.threshold_db,
            Modes.active_hits,
            Modes.inactive_misses);

    modesInitPLUTOSDR();
    debugPrintIioSettings();

    do {
        int ret = scanOnePass();
        if (ret < 0) {
            fprintf(stderr, "Scan error: %s (%d)\n", strerror(-ret), ret);
            break;
        }

        if (Modes.priority_mode && !Modes.exit) {
            ret = scanPriorityChannels();
            if (ret < 0) {
                fprintf(stderr, "Priority scan error: %s (%d)\n", strerror(-ret), ret);
                break;
            }
        }

        printSummary(mstime(), 0);
    } while (!Modes.exit && Modes.continuous);

    closeActiveChannels();
    if (Modes.summary_every_ms > 0) printSummary(mstime(), 1);

    if (Modes.chirp_csv_path && Modes.verbose) {
        fprintf(stderr, "CHIRP export complete: %d memories written to %s\n",
                Modes.chirp_export_count, Modes.chirp_csv_path);
    }

    modesCleanup();
    return 0;
}
