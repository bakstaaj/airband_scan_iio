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
 *   gcc -O2 -Wall -o airband_scan pluto_airband_scan.c -liio -lm
 *
 * Example:
 *   ./airband_scan --threshold 16 --gain -100
 *   ./airband_scan --start 118000000 --end 136975000 --step 25000
 */

#define _GNU_SOURCE
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

#define MODES_DEFAULT_RATE        2000000LL
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
#define DEFAULT_REPORT_HOLDOFF_MS 5000LL
#define DEFAULT_SETTLE_MS         15

typedef struct {
    float re;
    float im;
} cpxf;

struct air_channel {
    long long freq;
    long long last_report_ms;
    double last_score_db;
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

    struct air_channel *channels;
    int channel_count;
} Modes;

static long long mstime(void) {
    struct timeval tv;
    long long mst;
    gettimeofday(&tv, NULL);
    mst = ((long long)tv.tv_sec) * 1000;
    mst += tv.tv_usec / 1000;
    return mst;
}

static void intHandler(int sig) {
    MODES_NOTUSED(sig);
    Modes.exit = 1;
    Modes.stop = 1;
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

static void attr_write_ll(struct iio_channel *chn, const char *attr, long long val) {
    int ret = iio_channel_attr_write_longlong(chn, attr, val);
    if (ret < 0) die_iio(attr, ret);
}

static void attr_write_str(struct iio_channel *chn, const char *attr, const char *val) {
    int ret = iio_channel_attr_write(chn, attr, val);
    if (ret < 0) die_iio(attr, ret);
}

static void setCenterFrequency(long long center_hz) {
    attr_write_ll(Modes.lo_chn, "frequency", center_hz);
    Modes.freq = center_hz;
    if (Modes.settle_ms > 0) usleep((useconds_t)Modes.settle_ms * 1000U);
}

static void modesInitPLUTOSDR(void) {
    int device_count;

    printf("* Acquiring IIO context\n");
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

    printf("* Acquiring AD9361 streaming device cf-ad9361-lpc\n");
    Modes.dev = iio_context_find_device(Modes.ctx, "cf-ad9361-lpc");
    if (Modes.dev == NULL) {
        fprintf(stderr, "Could not find cf-ad9361-lpc.\n");
        exit(1);
    }

    printf("* Acquiring AD9361 phy device ad9361-phy\n");
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
    attr_write_ll(Modes.rx0_phy, "rf_bandwidth", Modes.rf_bandwidth);
    attr_write_ll(Modes.rx0_phy, "sampling_frequency", Modes.sample_rate);

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

    printf("* Initializing IIO streaming channels voltage0 / voltage1\n");
    Modes.rx0_i = iio_device_find_channel(Modes.dev, "voltage0", false);
    Modes.rx0_q = iio_device_find_channel(Modes.dev, "voltage1", false);
    if (!Modes.rx0_i || !Modes.rx0_q) {
        fprintf(stderr, "Could not find streaming channels voltage0 and voltage1.\n");
        exit(1);
    }
    iio_channel_enable(Modes.rx0_i);
    iio_channel_enable(Modes.rx0_q);

    printf("* Creating non-cyclic IIO RX buffer\n");
    Modes.rxbuf = iio_device_create_buffer(Modes.dev, (size_t)Modes.fft_size, false);
    if (!Modes.rxbuf) {
        perror("Could not create RX buffer");
        exit(1);
    }
}

static int buildChannelList(void) {
    long long f;
    int n = 0;
    if (Modes.channel_step <= 0 || Modes.scan_end < Modes.scan_start) return -1;

    for (f = Modes.scan_start; f <= Modes.scan_end; f += Modes.channel_step) n++;
    Modes.channels = calloc((size_t)n, sizeof(struct air_channel));
    if (!Modes.channels) return -1;

    n = 0;
    for (f = Modes.scan_start; f <= Modes.scan_end; f += Modes.channel_step) {
        Modes.channels[n].freq = f;
        Modes.channels[n].last_report_ms = 0;
        Modes.channels[n].last_score_db = 0.0;
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

static void printFrequency(long long hz, double score_db) {
    time_t now = time(NULL);
    struct tm tm_now;
    char tbuf[64];
    localtime_r(&now, &tm_now);
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", &tm_now);
    printf("%s  ACTIVE  %.3f MHz  score=%.1f dB\n", tbuf, (double)hz / 1000000.0, score_db);
    fflush(stdout);
}

static void discardBuffers(int count) {
    for (int i = 0; i < count; i++) {
        ssize_t ret = iio_buffer_refill(Modes.rxbuf);
        if (ret < 0) return;
    }
}

static int scanOnePass(void) {
    int n = Modes.fft_size;
    float *power = malloc((size_t)n * sizeof(float));
    if (!power) return -ENOMEM;

    double usable_bw = (double)Modes.sample_rate * Modes.usable_fraction;
    long long center_step = (long long)llround(usable_bw);
    long long half_usable = center_step / 2;
    long long center;
    long long center_offset = Modes.channel_step / 2; /* avoid DC spur landing exactly on a channel */
    int windows = 0;

    if (center_step <= 0) center_step = Modes.sample_rate / 2;

    for (center = Modes.scan_start + half_usable + center_offset;
         center <= Modes.scan_end + half_usable;
         center += center_step) {

        if (center - half_usable < Modes.scan_start) center = Modes.scan_start + half_usable + center_offset;
        if (center + half_usable > Modes.scan_end) center = Modes.scan_end - half_usable - center_offset;
        if (center < Modes.scan_start) center = (Modes.scan_start + Modes.scan_end) / 2;
        if (center > Modes.scan_end) center = (Modes.scan_start + Modes.scan_end) / 2;

        if (Modes.verbose) {
            fprintf(stderr, "Scanning window center %.6f MHz\n", (double)center / 1000000.0);
        }

        setCenterFrequency(center);
        discardBuffers(2);

        int ret = capturePowerSpectrum(power, n, Modes.fft_avg);
        if (ret < 0) {
            free(power);
            return ret;
        }

        float floor_power = percentile_copy(power, n, 0.35);
        long long low = center - half_usable;
        long long high = center + half_usable;
        long long now_ms = mstime();

        for (int c = 0; c < Modes.channel_count; c++) {
            long long f = Modes.channels[c].freq;
            if (f < low || f > high) continue;

            /* Avoid the residual DC spike at exactly zero IF. */
            if (llabs(f - center) < (Modes.channel_step / 2)) continue;

            double score = channelScoreDb(power, n, center, f, floor_power);
            Modes.channels[c].last_score_db = score;
            if (score >= Modes.threshold_db &&
                now_ms - Modes.channels[c].last_report_ms >= Modes.report_holdoff_ms) {
                printFrequency(f, score);
                Modes.channels[c].last_report_ms = now_ms;
            }
        }

        windows++;
        if (center + half_usable >= Modes.scan_end) break;
    }

    free(power);
    return windows;
}

static void showHelp(void) {
    printf(
        "Pluto+ airband activity scanner\n"
        "\n"
        "Options:\n"
        "  --start <Hz>          Start frequency, default 118000000\n"
        "  --end <Hz>            End frequency, default 136975000\n"
        "  --step <Hz>           Channel step, default 25000\n"
        "  --rate <Hz>           Sample rate, default 2000000\n"
        "  --bandwidth <Hz>      RF bandwidth, default = sample rate\n"
        "  --threshold <dB>      Detection threshold over noise, default 16\n"
        "  --detect-bw <Hz>      Detection bandwidth around channel, default 12000\n"
        "  --fft-size <N>        FFT size, power of 2, default 8192\n"
        "  --fft-avg <N>         FFT averages per tuning window, default 4\n"
        "  --gain <dB|-100>      Manual RX gain, or -100 for AGC; default -100\n"
        "  --settle-ms <N>       Retune settling delay, default 15\n"
        "  --holdoff-ms <N>      Repeat report holdoff per channel, default 5000\n"
        "  --once                Scan band once and exit\n"
        "  --verbose             Print tuning windows\n"
        "  --help                Show this help\n"
        "\n"
        "Example:\n"
        "  ./airband_scan --threshold 16 --gain -100\n"
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
        } else if (!strcmp(argv[j], "--end") && more) {
            Modes.scan_end = parse_ll(argv[++j], "end");
        } else if (!strcmp(argv[j], "--step") && more) {
            Modes.channel_step = parse_ll(argv[++j], "step");
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
        } else if (!strcmp(argv[j], "--once")) {
            Modes.continuous = 0;
        } else if (!strcmp(argv[j], "--verbose")) {
            Modes.verbose = 1;
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
    if (Modes.usable_fraction <= 0.0 || Modes.usable_fraction > 0.95) Modes.usable_fraction = DEFAULT_USABLE_FRACTION;

    signal(SIGINT, intHandler);
    signal(SIGTERM, intHandler);

    modesInit();
    if (buildChannelList() < 0) {
        fprintf(stderr, "Could not build channel list.\n");
        return 1;
    }

    printf("Airband scanner: %.3f-%.3f MHz, step %.3f kHz, %d channels\n",
           (double)Modes.scan_start / 1000000.0,
           (double)Modes.scan_end / 1000000.0,
           (double)Modes.channel_step / 1000.0,
           Modes.channel_count);
    printf("Rate %.3f MSPS, FFT %d x %d, threshold %.1f dB\n",
           (double)Modes.sample_rate / 1000000.0,
           Modes.fft_size,
           Modes.fft_avg,
           Modes.threshold_db);

    modesInitPLUTOSDR();

    do {
        int ret = scanOnePass();
        if (ret < 0) {
            fprintf(stderr, "Scan error: %s (%d)\n", strerror(-ret), ret);
            break;
        }
    } while (!Modes.exit && Modes.continuous);

    if (Modes.rxbuf) iio_buffer_destroy(Modes.rxbuf);
    if (Modes.ctx) iio_context_destroy(Modes.ctx);
    free(Modes.channels);
    free(Modes.maglut);
    free(Modes.magnitude);
    free(Modes.data);
    return 0;
}
