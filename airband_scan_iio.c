/*
   airband_scan_iio.c

   Local Pluto+/PlutoSDR VHF airband activity scanner using libiio.

   What it does:
     - Runs locally on the Pluto+/PlutoSDR Linux side.
     - Uses libiio local context.
     - Tunes RX LO across the airband in chunks.
     - Captures I/Q samples from cf-ad9361-lpc.
     - FFTs each chunk.
     - Reports channels whose peak rises above the chunk noise floor.

   Build:
     gcc -O2 -std=gnu99 -Wall -o airband_scan_iio airband_scan_iio.c -liio -lm

   Example:
     ./airband_scan_iio

   Example with more sensitivity:
     ./airband_scan_iio -t 12 -g 50

   Example for 8.33 kHz-ish spacing:
     ./airband_scan_iio -k 8.333 -t 14
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <stddef.h>

#if defined(__has_include)
#  if __has_include(<iio/iio.h>)
#    include <iio/iio.h>
#  else
#    include <iio.h>
#  endif
#else
#  include <iio.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEFAULT_START_HZ       118000000LL
#define DEFAULT_END_HZ         136975000LL
#define DEFAULT_STEP_HZ        25000LL
#define DEFAULT_SAMPLE_RATE_HZ 2000000LL
#define DEFAULT_RF_BW_HZ       1800000LL
#define DEFAULT_THRESHOLD_DB   16.0
#define DEFAULT_GAIN_DB        45.0
#define DEFAULT_FFT_SIZE       8192
#define DEFAULT_SETTLE_US      30000
#define REPORT_HOLDOFF_SEC     6

typedef struct {
    double r;
    double i;
} complex_d;

typedef struct {
    long long start_hz;
    long long end_hz;
    long long step_hz;
    long long sample_rate_hz;
    long long rf_bw_hz;
    double threshold_db;
    double gain_db;
    int fft_size;
    int settle_us;
    bool one_sweep;
    bool verbose;
    const char *agc_mode;
    const char *rf_port;
} config_t;

static volatile sig_atomic_t stop_requested = 0;

static void handle_sigint(int sig)
{
    (void)sig;
    stop_requested = 1;
}

static void iio_perror_msg(const char *msg, int ret)
{
    char errbuf[256];
    if (ret < 0) {
        iio_strerror(-ret, errbuf, sizeof(errbuf));
        fprintf(stderr, "%s: %s\n", msg, errbuf);
    } else {
        fprintf(stderr, "%s\n", msg);
    }
}

static bool is_power_of_two(int n)
{
    return n > 0 && ((n & (n - 1)) == 0);
}

static int cmp_double(const void *a, const void *b)
{
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void fft_forward(complex_d *x, int n)
{
    int i, j;

    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j) {
            complex_d tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        double wlen_r = cos(ang);
        double wlen_i = sin(ang);

        for (i = 0; i < n; i += len) {
            double w_r = 1.0;
            double w_i = 0.0;

            for (j = 0; j < len / 2; j++) {
                complex_d u = x[i + j];
                complex_d v;

                v.r = x[i + j + len / 2].r * w_r - x[i + j + len / 2].i * w_i;
                v.i = x[i + j + len / 2].r * w_i + x[i + j + len / 2].i * w_r;

                x[i + j].r = u.r + v.r;
                x[i + j].i = u.i + v.i;
                x[i + j + len / 2].r = u.r - v.r;
                x[i + j + len / 2].i = u.i - v.i;

                double next_r = w_r * wlen_r - w_i * wlen_i;
                double next_i = w_r * wlen_i + w_i * wlen_r;
                w_r = next_r;
                w_i = next_i;
            }
        }
    }
}

static long long mhz_to_hz(const char *s)
{
    double mhz = strtod(s, NULL);
    return (long long)llround(mhz * 1000000.0);
}

static long long khz_to_hz(const char *s)
{
    double khz = strtod(s, NULL);
    return (long long)llround(khz * 1000.0);
}

static void print_usage(const char *prog)
{
    printf(
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  -s MHz     Start frequency, default 118.000\n"
        "  -e MHz     End frequency, default 136.975\n"
        "  -k kHz     Channel step, default 25.000\n"
        "  -r MSPS    Sample rate, default 2.000\n"
        "  -b MHz     RF bandwidth, default 1.800\n"
        "  -t dB      Detection threshold above local floor, default 16\n"
        "  -g dB      Manual RX gain when AGC is manual, default 45\n"
        "  -a mode    AGC mode: manual, slow_attack, fast_attack. Default manual\n"
        "  -n N       FFT size, power of two, default 8192\n"
        "  -o         One sweep only\n"
        "  -v         Verbose\n"
        "  -h         Help\n"
        "\n"
        "Examples:\n"
        "  %s\n"
        "  %s -t 12 -g 50\n"
        "  %s -k 8.333 -t 14\n",
        prog, prog, prog, prog
    );
}

static int set_attr_ll(struct iio_channel *ch, const char *attr, long long value)
{
    int ret = iio_channel_attr_write_longlong(ch, attr, value);
    if (ret < 0) {
        char msg[160];
        snprintf(msg, sizeof(msg), "Failed writing %s=%lld", attr, value);
        iio_perror_msg(msg, ret);
    }
    return ret;
}

static int set_attr_str(struct iio_channel *ch, const char *attr, const char *value)
{
    int ret = iio_channel_attr_write_string(ch, attr, value);
    if (ret < 0) {
        char msg[160];
        snprintf(msg, sizeof(msg), "Failed writing %s=%s", attr, value);
        iio_perror_msg(msg, ret);
    }
    return ret;
}

static int set_attr_double_as_str(struct iio_channel *ch, const char *attr, double value)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f", value);
    int ret = iio_channel_attr_write(ch, attr, buf);
    if (ret < 0) {
        char msg[160];
        snprintf(msg, sizeof(msg), "Failed writing %s=%s", attr, buf);
        iio_perror_msg(msg, ret);
    }
    return ret;
}

static int configure_rx(
    struct iio_device *phy,
    struct iio_channel **rx_phy_ch,
    struct iio_channel **rx_lo_ch,
    const config_t *cfg
) {
    *rx_phy_ch = iio_device_find_channel(phy, "voltage0", false);
    if (!*rx_phy_ch) {
        fprintf(stderr, "Could not find RX phy channel ad9361-phy/voltage0\n");
        return -1;
    }

    *rx_lo_ch = iio_device_find_channel(phy, "altvoltage0", true);
    if (!*rx_lo_ch) {
        fprintf(stderr, "Could not find RX LO channel ad9361-phy/altvoltage0\n");
        return -1;
    }

    int ret;

    ret = iio_channel_attr_write_string(*rx_phy_ch, "rf_port_select", cfg->rf_port);
    if (ret < 0 && cfg->verbose) {
        fprintf(stderr, "Warning: could not set rf_port_select=%s; continuing\n", cfg->rf_port);
    }

    ret = set_attr_ll(*rx_phy_ch, "rf_bandwidth", cfg->rf_bw_hz);
    if (ret < 0) return ret;

    ret = set_attr_ll(*rx_phy_ch, "sampling_frequency", cfg->sample_rate_hz);
    if (ret < 0) return ret;

    ret = set_attr_str(*rx_phy_ch, "gain_control_mode", cfg->agc_mode);
    if (ret < 0) return ret;

    if (strcmp(cfg->agc_mode, "manual") == 0) {
        ret = set_attr_double_as_str(*rx_phy_ch, "hardwaregain", cfg->gain_db);
        if (ret < 0) return ret;
    }

    return 0;
}

static int tune_rx(struct iio_channel *rx_lo_ch, long long center_hz)
{
    return set_attr_ll(rx_lo_ch, "frequency", center_hz);
}

static int signed_bin_index(int bin, int n)
{
    return (bin <= n / 2) ? bin : bin - n;
}

static int offset_hz_to_bin(double offset_hz, double sample_rate_hz, int n)
{
    long signed_bin = lround((offset_hz / sample_rate_hz) * (double)n);

    while (signed_bin < -n / 2)
        signed_bin += n;
    while (signed_bin > n / 2)
        signed_bin -= n;

    if (signed_bin < 0)
        return n + signed_bin;
    return signed_bin;
}

static double median_noise_floor_db(const double *spectrum_db, int n, int dc_skip_bins)
{
    double *tmp = malloc(sizeof(double) * (size_t)n);
    if (!tmp)
        return -300.0;

    int count = 0;

    for (int k = 0; k < n; k++) {
        int sb = signed_bin_index(k, n);
        if (abs(sb) <= dc_skip_bins)
            continue;

        tmp[count++] = spectrum_db[k];
    }

    if (count <= 0) {
        free(tmp);
        return -300.0;
    }

    qsort(tmp, (size_t)count, sizeof(double), cmp_double);
    double median = tmp[count / 2];
    free(tmp);
    return median;
}

static double channel_peak_db(
    const double *spectrum_db,
    int n,
    double sample_rate_hz,
    double offset_hz,
    double channel_bw_hz,
    int dc_skip_bins
) {
    double bin_hz = sample_rate_hz / (double)n;
    int half_bins = (int)ceil((channel_bw_hz * 0.5) / bin_hz);
    if (half_bins < 1)
        half_bins = 1;

    int center_bin = offset_hz_to_bin(offset_hz, sample_rate_hz, n);
    double peak = -300.0;

    for (int dk = -half_bins; dk <= half_bins; dk++) {
        int b = center_bin + dk;
        while (b < 0) b += n;
        while (b >= n) b -= n;

        int sb = signed_bin_index(b, n);
        if (abs(sb) <= dc_skip_bins)
            continue;

        if (spectrum_db[b] > peak)
            peak = spectrum_db[b];
    }

    return peak;
}

static int capture_fft_spectrum(
    struct iio_buffer *rxbuf,
    struct iio_channel *rx_i,
    struct iio_channel *rx_q,
    complex_d *fft_buf,
    double *spectrum_db,
    int n
) {
    ssize_t nbytes = iio_buffer_refill(rxbuf);
    if (nbytes < 0) {
        iio_perror_msg("iio_buffer_refill failed", (int)nbytes);
        return -1;
    }

    char *pi = (char *)iio_buffer_first(rxbuf, rx_i);
    char *pq = (char *)iio_buffer_first(rxbuf, rx_q);
    char *end = (char *)iio_buffer_end(rxbuf);
    ptrdiff_t step = iio_buffer_step(rxbuf);

    double mean_i = 0.0;
    double mean_q = 0.0;
    int count = 0;

    char *ti = pi;
    char *tq = pq;

    for (int i = 0; i < n && ti < end && tq < end; i++, ti += step, tq += step) {
        int16_t si = *(int16_t *)ti;
        int16_t sq = *(int16_t *)tq;

        double di = (double)si / 32768.0;
        double dq = (double)sq / 32768.0;

        fft_buf[i].r = di;
        fft_buf[i].i = dq;

        mean_i += di;
        mean_q += dq;
        count++;
    }

    if (count < n) {
        fprintf(stderr, "Short IIO buffer: got %d complex samples, expected %d\n", count, n);
        return -1;
    }

    mean_i /= (double)n;
    mean_q /= (double)n;

    for (int i = 0; i < n; i++) {
        double w = 0.5 - 0.5 * cos((2.0 * M_PI * (double)i) / (double)(n - 1));
        fft_buf[i].r = (fft_buf[i].r - mean_i) * w;
        fft_buf[i].i = (fft_buf[i].i - mean_q) * w;
    }

    fft_forward(fft_buf, n);

    for (int k = 0; k < n; k++) {
        double p = fft_buf[k].r * fft_buf[k].r + fft_buf[k].i * fft_buf[k].i;
        spectrum_db[k] = 10.0 * log10(p + 1e-30);
    }

    return 0;
}

static void print_active(
    long long freq_hz,
    double level_db,
    double floor_db,
    double margin_db
) {
    time_t now = time(NULL);
    struct tm tm_now;
    localtime_r(&now, &tm_now);

    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", &tm_now);

    printf(
        "%s  ACTIVE  %.6f MHz   level %.1f dB   floor %.1f dB   margin %.1f dB\n",
        tbuf,
        (double)freq_hz / 1000000.0,
        level_db,
        floor_db,
        margin_db
    );

    fflush(stdout);
}

int main(int argc, char **argv)
{
    config_t cfg;
    cfg.start_hz = DEFAULT_START_HZ;
    cfg.end_hz = DEFAULT_END_HZ;
    cfg.step_hz = DEFAULT_STEP_HZ;
    cfg.sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ;
    cfg.rf_bw_hz = DEFAULT_RF_BW_HZ;
    cfg.threshold_db = DEFAULT_THRESHOLD_DB;
    cfg.gain_db = DEFAULT_GAIN_DB;
    cfg.fft_size = DEFAULT_FFT_SIZE;
    cfg.settle_us = DEFAULT_SETTLE_US;
    cfg.one_sweep = false;
    cfg.verbose = false;
    cfg.agc_mode = "manual";
    cfg.rf_port = "A_BALANCED";

    int opt;
    while ((opt = getopt(argc, argv, "s:e:k:r:b:t:g:a:n:ovh")) != -1) {
        switch (opt) {
            case 's':
                cfg.start_hz = mhz_to_hz(optarg);
                break;
            case 'e':
                cfg.end_hz = mhz_to_hz(optarg);
                break;
            case 'k':
                cfg.step_hz = khz_to_hz(optarg);
                break;
            case 'r':
                cfg.sample_rate_hz = mhz_to_hz(optarg);
                break;
            case 'b':
                cfg.rf_bw_hz = mhz_to_hz(optarg);
                break;
            case 't':
                cfg.threshold_db = strtod(optarg, NULL);
                break;
            case 'g':
                cfg.gain_db = strtod(optarg, NULL);
                break;
            case 'a':
                cfg.agc_mode = optarg;
                break;
            case 'n':
                cfg.fft_size = atoi(optarg);
                break;
            case 'o':
                cfg.one_sweep = true;
                break;
            case 'v':
                cfg.verbose = true;
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return 0;
        }
    }

    if (cfg.start_hz >= cfg.end_hz) {
        fprintf(stderr, "Start frequency must be below end frequency\n");
        return 1;
    }

    if (cfg.step_hz <= 0) {
        fprintf(stderr, "Channel step must be positive\n");
        return 1;
    }

    if (!is_power_of_two(cfg.fft_size)) {
        fprintf(stderr, "FFT size must be a power of two\n");
        return 1;
    }

    if (cfg.rf_bw_hz > cfg.sample_rate_hz) {
        fprintf(stderr, "RF bandwidth should not exceed sample rate\n");
        return 1;
    }

    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    printf("Airband IIO scanner starting\n");
    printf("Range: %.6f to %.6f MHz, step %.3f kHz\n",
           (double)cfg.start_hz / 1000000.0,
           (double)cfg.end_hz / 1000000.0,
           (double)cfg.step_hz / 1000.0);
    printf("Sample rate: %.3f MSPS, RF BW: %.3f MHz, FFT: %d, threshold: %.1f dB\n",
           (double)cfg.sample_rate_hz / 1000000.0,
           (double)cfg.rf_bw_hz / 1000000.0,
           cfg.fft_size,
           cfg.threshold_db);
    printf("Press Ctrl-C to stop.\n\n");

    struct iio_context *ctx = iio_create_local_context();
    if (!ctx) {
        fprintf(stderr, "Could not create local IIO context\n");
        return 1;
    }

    struct iio_device *phy = iio_context_find_device(ctx, "ad9361-phy");
    if (!phy) {
        fprintf(stderr, "Could not find IIO device ad9361-phy\n");
        iio_context_destroy(ctx);
        return 1;
    }

    struct iio_device *rxdev = iio_context_find_device(ctx, "cf-ad9361-lpc");
    if (!rxdev) {
        fprintf(stderr, "Could not find IIO RX stream device cf-ad9361-lpc\n");
        iio_context_destroy(ctx);
        return 1;
    }

    struct iio_channel *rx_phy_ch = NULL;
    struct iio_channel *rx_lo_ch = NULL;

    if (configure_rx(phy, &rx_phy_ch, &rx_lo_ch, &cfg) < 0) {
        iio_context_destroy(ctx);
        return 1;
    }

    struct iio_channel *rx_i = iio_device_find_channel(rxdev, "voltage0", false);
    struct iio_channel *rx_q = iio_device_find_channel(rxdev, "voltage1", false);

    if (!rx_i || !rx_q) {
        fprintf(stderr, "Could not find RX stream voltage0/voltage1 channels\n");
        iio_context_destroy(ctx);
        return 1;
    }

    iio_channel_enable(rx_i);
    iio_channel_enable(rx_q);

    struct iio_buffer *rxbuf = iio_device_create_buffer(rxdev, (size_t)cfg.fft_size, false);
    if (!rxbuf) {
        fprintf(stderr, "Could not create RX buffer\n");
        iio_context_destroy(ctx);
        return 1;
    }

    complex_d *fft_buf = calloc((size_t)cfg.fft_size, sizeof(complex_d));
    double *spectrum_db = calloc((size_t)cfg.fft_size, sizeof(double));

    if (!fft_buf || !spectrum_db) {
        fprintf(stderr, "Out of memory\n");
        free(fft_buf);
        free(spectrum_db);
        iio_buffer_destroy(rxbuf);
        iio_context_destroy(ctx);
        return 1;
    }

    int channel_count = (int)(((cfg.end_hz - cfg.start_hz) / cfg.step_hz) + 1);
    time_t *last_report = calloc((size_t)channel_count, sizeof(time_t));
    if (!last_report) {
        fprintf(stderr, "Out of memory\n");
        free(fft_buf);
        free(spectrum_db);
        iio_buffer_destroy(rxbuf);
        iio_context_destroy(ctx);
        return 1;
    }

    double sample_rate = (double)cfg.sample_rate_hz;
    double analysis_half_hz = sample_rate * 0.45;
    double chunk_step_hz = sample_rate * 0.75;
    double channel_bw_hz = fmin((double)cfg.step_hz * 0.80, 20000.0);
    if (channel_bw_hz < 5000.0)
        channel_bw_hz = (double)cfg.step_hz * 0.70;

    int dc_skip_bins = (int)ceil(3000.0 / (sample_rate / (double)cfg.fft_size));
    if (dc_skip_bins < 2)
        dc_skip_bins = 2;

    while (!stop_requested) {
        double first_center = (double)cfg.start_hz + analysis_half_hz - ((double)cfg.step_hz * 0.5);

        for (double center_d = first_center;
             center_d - analysis_half_hz <= (double)cfg.end_hz && !stop_requested;
             center_d += chunk_step_hz) {

            long long center_hz = (long long)llround(center_d);

            if (cfg.verbose) {
                printf("Tuning %.6f MHz\n", (double)center_hz / 1000000.0);
            }

            if (tune_rx(rx_lo_ch, center_hz) < 0) {
                fprintf(stderr,
                        "Tuning failed at %.6f MHz. If this is near 118 MHz, your Pluto firmware/hardware may not support VHF airband.\n",
                        (double)center_hz / 1000000.0);
                stop_requested = 1;
                break;
            }

            usleep((useconds_t)cfg.settle_us);

            /*
               Discard one buffer after tuning so we do not analyze stale samples.
            */
            ssize_t discard = iio_buffer_refill(rxbuf);
            if (discard < 0) {
                iio_perror_msg("Discard refill failed", (int)discard);
                stop_requested = 1;
                break;
            }

            if (capture_fft_spectrum(rxbuf, rx_i, rx_q, fft_buf, spectrum_db, cfg.fft_size) < 0) {
                stop_requested = 1;
                break;
            }

            double floor_db = median_noise_floor_db(spectrum_db, cfg.fft_size, dc_skip_bins);
            time_t now = time(NULL);

            for (int ch = 0; ch < channel_count; ch++) {
                long long freq_hz = cfg.start_hz + ((long long)ch * cfg.step_hz);
                if (freq_hz > cfg.end_hz)
                    continue;

                double offset_hz = (double)freq_hz - (double)center_hz;

                if (fabs(offset_hz) > analysis_half_hz)
                    continue;

                double level_db = channel_peak_db(
                    spectrum_db,
                    cfg.fft_size,
                    sample_rate,
                    offset_hz,
                    channel_bw_hz,
                    dc_skip_bins
                );

                double margin_db = level_db - floor_db;

                if (margin_db >= cfg.threshold_db) {
                    if (now - last_report[ch] >= REPORT_HOLDOFF_SEC) {
                        print_active(freq_hz, level_db, floor_db, margin_db);
                        last_report[ch] = now;
                    }
                }
            }
        }

        if (cfg.one_sweep)
            break;
    }

    printf("\nScanner stopped.\n");

    free(last_report);
    free(fft_buf);
    free(spectrum_db);
    iio_buffer_destroy(rxbuf);
    iio_context_destroy(ctx);

    return 0;
}