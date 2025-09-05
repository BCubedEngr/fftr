/**
 * Note that this test is not distributed with the package so the package
 * doesn't take on the GPL license.
 **/

#include <fftw3.h>
#include <gtest/gtest.h>
#include <chrono>
#include <complex>
#include <concepts>
#include <fftr/fft.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using rustfft::FFT;

/*
Tests to write
x compare result to fftw
check inplace vs out of place computation
x check normalize
 */

using test_types = ::testing::Types<float, std::complex<float>, double, std::complex<double>>;

template <class T, class U>
void compare_vectors(const T& v1, const U& v2, double rtol = 1e-6, double atol = 1e-6) {
    using value_type = typename T::value_type;
    EXPECT_EQ(v1.size(), v2.size());
    std::vector<std::tuple<size_t, value_type, value_type>> errors;
    size_t error_count = 0;
    for (size_t index = 0; index < v1.size(); index++) {
        const auto a = v1[index];
        const auto b = v2[index];
        if constexpr (std::integral<value_type>) {
            if (a != b) {
                error_count += 1;
                if (errors.size() < 10)
                    errors.emplace_back(std::tuple{index, a, b});
            }
        }
        else {
            auto diff = std::abs(a - b);
            if (diff > rtol && diff / std::abs(b) > atol) {
                error_count += 1;
                if (errors.size() < 10)
                    errors.emplace_back(std::tuple{index, a, b});
            }
        }
    }
    if (error_count > 0) {
        // Make the error Message
        std::stringstream ss;
        ss << "Vectors differ in " << error_count << " locations\n";
        for (const auto& [index, a, b] : errors) {
            ss << "\tindex=" << index << ", a=" << a << ", b=" << b << ", diff=" << a - b
               << std::endl;
        }

        EXPECT_TRUE(error_count == 0) << ss.str();
    }
}

void check_complex_fft_result(size_t N) {
    std::vector<std::complex<float>> in(N);
    std::vector<std::complex<float>> out1w(N);
    std::vector<std::complex<float>> out2w(N);
    std::vector<std::complex<float>> out1r(N);
    std::vector<std::complex<float>> out2r(N);
    // Use the same seed every time
    std::srand(N);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distrib(-0.5, 0.5);
    for (size_t i = 0; i < in.size(); i++) {
        in[i] = std::complex<float>(distrib(generator), distrib(generator));
    }
    // FFTW does not normalize so we don't either here
    FFT<std::complex<float>> fft(N, false);
    fft.execute_forward(out1r, in);
    fft.execute_inverse(out2r, out1r);

    fftwf_plan p_f, p_i;
    p_f = fftwf_plan_dft_1d(static_cast<int>(N),
                            reinterpret_cast<fftwf_complex*>(in.data()),
                            reinterpret_cast<fftwf_complex*>(out1w.data()),
                            FFTW_FORWARD,
                            FFTW_ESTIMATE);
    p_i = fftwf_plan_dft_1d(static_cast<int>(N),
                            reinterpret_cast<fftwf_complex*>(out1w.data()),
                            reinterpret_cast<fftwf_complex*>(out2w.data()),
                            FFTW_BACKWARD,
                            FFTW_ESTIMATE);
    fftwf_execute(p_f);
    fftwf_execute(p_i);
    fftwf_destroy_plan(p_f);
    fftwf_destroy_plan(p_i);

    compare_vectors(out1w, out1r, 6e-5, 6e-5);
    compare_vectors(out2w, out2r, 6e-5, 6e-5 * N);
}

void check_real_fft_result(size_t N) {
    size_t halfN = N / 2 + 1;
    std::vector<float> in(N);
    std::vector<std::complex<float>> out1w(halfN);
    std::vector<float> out2w(N);
    std::vector<std::complex<float>> out1r(halfN);
    std::vector<float> out2r(N);
    // Use the same seed every time
    std::srand(N);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distrib(-0.5, 0.5);
    for (size_t i = 0; i < in.size(); i++) {
        in[i] = distrib(generator);
    }
    const std::vector<float> in_c(in);
    FFT<float> fft(N, false);
    fft.execute_forward(out1r, in_c);
    fft.execute_inverse(out2r, out1r);

    fftwf_plan p_f, p_i;
    p_f = fftwf_plan_dft_r2c_1d(static_cast<int>(N),
                                in.data(),
                                reinterpret_cast<fftwf_complex*>(out1w.data()),
                                FFTW_ESTIMATE);
    p_i = fftwf_plan_dft_c2r_1d(static_cast<int>(N),
                                reinterpret_cast<fftwf_complex*>(out1w.data()),
                                out2w.data(),
                                FFTW_ESTIMATE);
    fftwf_execute(p_f);
    //  The c2r destroys the input vector
    const std::vector<std::complex<float>> out1w_c(out1w);
    fftwf_execute(p_i);
    fftwf_destroy_plan(p_f);
    fftwf_destroy_plan(p_i);

    compare_vectors(out1w_c, out1r, 5e-5, 5e-5);
    compare_vectors(out2w, out2r, 5e-5, 5e-5 * N);
}

template <class T>
void check_normalize(size_t N) {
    using complex_t = rustfft::ComplexType<T>;
    using real_t = typename complex_t::value_type;
    std::vector<complex_t> in(N);
    std::vector<T> out1(N);
    std::vector<T> out2(N);
    float scale = 1.0 / static_cast<float>(N);
    for (size_t i = 0; i < in.size(); i++) {
        float index = static_cast<float>(i);
        in[i] = std::complex<float>(index / scale - 0.5, -3 * index / scale + 1.5);
    }
    // For real data the first and last elements don't have an imaginary part
    if constexpr (std::same_as<T, real_t>) {
        in.resize(N / 2 + 1);
        in[0] = {1, 0};
        in[N / 2] = {-1, 0};
    }

    FFT<T> fft_norm(N, true);
    FFT<T> fft_no(N, false);
    fft_norm.execute_inverse(out1, in);
    fft_no.execute_inverse(out2, in);

    for (auto& v : out2) {
        v *= scale;
    }
    compare_vectors(out1, out2);
}

template <class T>
void check_inplace(size_t N) {
    using complex_t = rustfft::ComplexType<T>;
    using real_t = typename complex_t::value_type;
    std::vector<T> in(N);
    std::vector<T> time1(N);
    std::vector<T> time2(N);

    float scale = 1.0 / static_cast<float>(N);
    for (size_t i = 0; i < in.size(); i++) {
        float index = static_cast<float>(i);
        if constexpr (std::same_as<T, real_t>)
            in[i] = index / scale - 0.5;
        else
            in[i] = complex_t(index / scale - 0.5, -3 * index / scale + 1.5);
    }

    FFT<T> fft(N, false);
    const std::vector<T> in_c(in);
    if constexpr (std::same_as<T, complex_t>) {
        std::vector<complex_t> freq1(N);
        std::vector<complex_t> freq2(in);
        fft.execute_forward(freq1, in_c);
        fft.execute_forward(freq2);
        compare_vectors(freq1, freq2);
        fft.execute_inverse(time1, freq1);
        fft.execute_inverse(time2, freq2);
        compare_vectors(time1, time2);
    }
    else {
        std::vector<complex_t> freq1(N / 2 + 1);
        std::vector<complex_t> freq2(N / 2 + 1);
        fft.execute_forward(freq1, in_c);
        fft.execute_forward(freq2, in);
        compare_vectors(freq1, freq2);
        const std::vector<complex_t> freq1_c(freq1);
        fft.execute_inverse(time1, freq1_c);
        fft.execute_inverse(time2, freq2);
        compare_vectors(time1, time2);
    }
}

template <class T>
class FFTTestSuite : public ::testing::Test {};
TYPED_TEST_SUITE(FFTTestSuite, test_types);

TEST(FFTTestSuite, ComplexTest) {
    for (size_t N = 2; N < 32; N++) {
        check_complex_fft_result(N);
    }
    for (size_t N = 32; N <= 65536; N *= 2) {
        check_complex_fft_result(N);
    }
}
TEST(FFTTestSuite, RealTest) {
    for (size_t N = 2; N < 32; N++) {
        check_real_fft_result(N);
    }
    for (size_t N = 32; N <= 65536; N *= 2) {
        check_real_fft_result(N);
    }
}

TYPED_TEST(FFTTestSuite, Normalize) {
    check_normalize<TypeParam>(37);
    check_normalize<TypeParam>(1024);
}

TYPED_TEST(FFTTestSuite, CheckInPlace) {
    check_inplace<TypeParam>(37);
    check_inplace<TypeParam>(2048);
}

template <class T>
class Wrapper {
   public:
    Wrapper(size_t size) {
        _fft = FFT<T>(size, false);
    }

   private:
    FFT<T> _fft = FFT<T>(1, false);
};

TYPED_TEST(FFTTestSuite, DoubleFree) {
    // There has been a problem of double freeing when we have multiple creates and frees
    // Make sure that these go out of scope properly
    {
        Wrapper<std::complex<float>> w1(1024);
        w1 = Wrapper<std::complex<float>>(2048);
    }
    {
        FFT<TypeParam> fft1(1024, false);
        fft1 = FFT<TypeParam>(2048, false);
    }
    {
        void* a1 = rustfft::detail::create_fft_plan<TypeParam>();
        void* a2 = rustfft::detail::create_fft_plan<TypeParam>();
        rustfft::detail::free_fft_plan<TypeParam>(a1);
        rustfft::detail::free_fft_plan<TypeParam>(a2);
    }
}

void benchmark_complex_fft_result(size_t N, size_t iterations) {
    std::vector<std::complex<float>> in(N);
    std::vector<std::complex<float>> out(N);
    // Use the same seed every time
    std::srand(N);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distrib(-0.5, 0.5);
    for (size_t i = 0; i < in.size(); i++) {
        in[i] = std::complex<float>(distrib(generator), distrib(generator));
    }
    // FFTW does not normalize so we don't either here
    FFT<std::complex<float>> fft(N, false);
    auto start_time = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; i++)
        fft.execute_forward(out, in);
    auto end_time = std::chrono::steady_clock::now();
    auto time_diff = end_time - start_time;
    std::cout << "Complex " << N << " FFTR - time (ms) = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_diff).count()
              << std::endl;

    fftwf_plan p_f;
    p_f = fftwf_plan_dft_1d(static_cast<int>(N),
                            reinterpret_cast<fftwf_complex*>(in.data()),
                            reinterpret_cast<fftwf_complex*>(out.data()),
                            FFTW_FORWARD,
                            FFTW_ESTIMATE);
    start_time = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; i++)
        fftwf_execute(p_f);
    end_time = std::chrono::steady_clock::now();
    time_diff = end_time - start_time;
    std::cout << "Complex " << N << " FFTW - time (ms) = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(time_diff).count()
              << std::endl;
    std::cout << std::endl;

    fftwf_destroy_plan(p_f);
}

/*TYPED_TEST(FFTTestSuite, Benchmark) {
    for (size_t N = 32; N <= 65536 * 32; N *= 2) {
        benchmark_complex_fft_result(N, 5000 * 65536 / N);
    }
}*/

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
