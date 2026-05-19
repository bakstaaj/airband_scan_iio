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
 * v4 note: --verbose prints each actual tuned LO center frequency and
 * per-window max score; end-of-band retune logic avoids repeating one center.
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

    double usable_bw_d = (double)Modes.sample_rate * Modes.usable_fraction;
    long long usable_span = (long long)llround(usable_bw_d);
    long long band_span = Modes.scan_end - Modes.scan_start;
    long long half_usable;
    long long base_center;
    int windows = 0;

    if (usable_span <= 0) usable_span = Modes.sample_rate / 2;
    if (usable_span <= 0) usable_span = Modes.channel_step;
    if (band_span > 0 && usable_span > band_span) usable_span = band_span;
    if (usable_span < Modes.channel_step) usable_span = Modes.channel_step;

    half_usable = usable_span / 2;
    if (half_usable <= 0) half_usable = Modes.channel_step / 2;

    /*
     * Walk the band in non-overlapping usable-span windows.  Earlier versions
     * modified the for-loop variable while clamping the final window, which
     * could repeatedly tune the same last frequency forever.  This loop keeps
     * the requested base center separate from the clamped/tuned center.
     */
    base_center = Modes.scan_start + half_usable;
    while (!Modes.exit) {
        long long center = base_center;
        long long tuned_center;
        long long low;
        long long high;
        long long actual_low;
        long long actual_high;
        int channels_in_window = 0;
        int channels_tested = 0;
        double max_score = -999.0;
        long long max_score_freq = 0;
        int ret;

        if (center - half_usable < Modes.scan_start) {
            center = Modes.scan_start + half_usable;
        }
        if (center + half_usable > Modes.scan_end) {
            center = Modes.scan_end - half_usable;
        }
        if (center < Modes.scan_start || center > Modes.scan_end || half_usable <= 0) {
            center = (Modes.scan_start + Modes.scan_end) / 2;
        }

        low = center - half_usable;
        high = center + half_usable;
        if (low < Modes.scan_start) low = Modes.scan_start;
        if (high > Modes.scan_end) high = Modes.scan_end;

        if (Modes.verbose) {
            fprintf(stderr,
                    "TUNING center=%.6f MHz  window=%.6f-%.6f MHz\n",
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

        ret = capturePowerSpectrum(power, n, Modes.fft_avg);
        if (ret < 0) {
            free(power);
            return ret;
        }

        float floor_power = percentile_copy(power, n, 0.35);
        long long now_ms = mstime();

        for (int c = 0; c < Modes.channel_count; c++) {
            long long f = Modes.channels[c].freq;
            if (f < actual_low || f > actual_high) continue;
            channels_in_window++;

            /* Avoid the residual DC spike at exactly zero IF. */
            if (llabs(f - tuned_center) < (Modes.channel_step / 2)) continue;
            channels_tested++;

            double score = channelScoreDb(power, n, tuned_center, f, floor_power);
            Modes.channels[c].last_score_db = score;
            if (score > max_score) {
                max_score = score;
                max_score_freq = f;
            }
            if (score >= Modes.threshold_db &&
                now_ms - Modes.channels[c].last_report_ms >= Modes.report_holdoff_ms) {
                printFrequency(f, score);
                Modes.channels[c].last_report_ms = now_ms;
            }
        }

        if (Modes.verbose) {
            if (channels_tested > 0) {
                fprintf(stderr,
                        "TUNED  center=%.6f MHz  actual-window=%.6f-%.6f MHz  channels=%d tested=%d  max=%.1f dB @ %.3f MHz\n",
                        (double)tuned_center / 1000000.0,
                        (double)actual_low / 1000000.0,
                        (double)actual_high / 1000000.0,
                        channels_in_window,
                        channels_tested,
                        max_score,
                        (double)max_score_freq / 1000000.0);
            } else {
                fprintf(stderr,
                        "TUNED  center=%.6f MHz  actual-window=%.6f-%.6f MHz  channels=%d tested=%d\n",
                        (double)tuned_center / 1000000.0,
                        (double)actual_low / 1000000.0,
                        (double)actual_high / 1000000.0,
                        channels_in_window,
                        channels_tested);
            }
            fflush(stderr);
        }

        windows++;
        if (center + half_usable >= Modes.scan_end) break;

        base_center += usable_span;
        if (base_center > Modes.scan_end + half_usable) break;
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
        "  --rate <Hz>           Sample rate, default 2400000\n"
        "  --bandwidth <Hz>      RF bandwidth, default = sample rate\n"
        "  --threshold <dB>      Detection threshold over noise, default 16\n"
        "  --detect-bw <Hz>      Detection bandwidth around channel, default 12000\n"
        "  --fft-size <N>        FFT size, power of 2, default 8192\n"
        "  --fft-avg <N>         FFT averages per tuning window, default 4\n"
        "  --gain <dB|-100>      Manual RX gain, or -100 for AGC; default -100\n"
        "  --settle-ms <N>       Retune settling delay, default 15\n"
        "  --holdoff-ms <N>      Repeat report holdoff per channel, default 5000\n"
        "  --once                Scan band once and exit\n"
        "  --verbose             Print each tuned LO center, scan window, and max score\n"
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
