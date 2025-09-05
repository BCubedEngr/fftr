#ifndef INCLUDE_FFTR_FFT_HPP_
#define INCLUDE_FFTR_FFT_HPP_
#include <algorithm>
#include <complex>
#include <concepts>
#include <iostream>
#include <sstream>
#include <vector>

namespace rustfft {

namespace detail {
extern "C" {
void* create_fft_fc32();
void* create_r2c_fft_fc32();
void free_fft_fc32(void* plan);
void free_r2c_fft_fc32(void* plan);
void execute_fft_fc32(void* plan,
                      std::complex<float>* output,
                      size_t fft_size,
                      bool forward,
                      bool normalize);
void execute_r2c_fft_fc32(void* plan, std::complex<float>* output, float* input, size_t fft_size);
void execute_c2r_fft_fc32(void* plan,
                          float* output,
                          std::complex<float>* input,
                          size_t fft_size,
                          bool normalize);
void* create_fft_fc64();
void* create_r2c_fft_fc64();
void free_fft_fc64(void* plan);
void free_r2c_fft_fc64(void* plan);
void execute_fft_fc64(void* plan,
                      std::complex<double>* output,
                      size_t fft_size,
                      bool forward,
                      bool normalize);
void execute_r2c_fft_fc64(void* plan, std::complex<double>* output, double* input, size_t fft_size);
void execute_c2r_fft_fc64(void* plan,
                          double* output,
                          std::complex<double>* input,
                          size_t fft_size,
                          bool normalize);
}

template <std::same_as<float> time_t>
void* create_fft_plan() {
    return create_r2c_fft_fc32();
}

template <std::same_as<std::complex<float>> time_t>
void* create_fft_plan() {
    return create_fft_fc32();
}

template <std::same_as<float> time_t>
void free_fft_plan(void* plan) {
    free_r2c_fft_fc32(plan);
}

template <std::same_as<std::complex<float>> time_t>
void free_fft_plan(void* plan) {
    free_fft_fc32(plan);
}

template <std::same_as<std::complex<float>> freq_t>
void execute_fft(void* plan, freq_t* output, size_t fft_size, bool direction, bool normalize) {
    execute_fft_fc32(plan, output, fft_size, direction, normalize);
}

template <std::same_as<float> time_t>
void execute_r2c_fft(void* plan, std::complex<time_t>* output, time_t* input, size_t fft_size) {
    execute_r2c_fft_fc32(plan, output, input, fft_size);
}

template <std::same_as<float> time_t>
void execute_c2r_fft(void* plan,
                     time_t* output,
                     std::complex<time_t>* input,
                     size_t fft_size,
                     bool normalize) {
    execute_c2r_fft_fc32(plan, output, input, fft_size, normalize);
}

template <std::same_as<double> time_t>
void* create_fft_plan() {
    return create_r2c_fft_fc64();
}

template <std::same_as<std::complex<double>> time_t>
void* create_fft_plan() {
    return create_fft_fc64();
}

template <std::same_as<double> time_t>
void free_fft_plan(void* plan) {
    free_r2c_fft_fc64(plan);
}

template <std::same_as<std::complex<double>> time_t>
void free_fft_plan(void* plan) {
    free_fft_fc64(plan);
}

template <std::same_as<std::complex<double>> freq_t>
void execute_fft(void* plan, freq_t* output, size_t fft_size, bool direction, bool normalize) {
    execute_fft_fc64(plan, output, fft_size, direction, normalize);
}

template <std::same_as<double> time_t>
void execute_r2c_fft(void* plan, std::complex<time_t>* output, time_t* input, size_t fft_size) {
    execute_r2c_fft_fc64(plan, output, input, fft_size);
}

template <std::same_as<double> time_t>
void execute_c2r_fft(void* plan,
                     time_t* output,
                     std::complex<time_t>* input,
                     size_t fft_size,
                     bool normalize) {
    execute_c2r_fft_fc64(plan, output, input, fft_size, normalize);
}

}  // namespace detail

template <class T>
concept Complex = std::floating_point<typename T::value_type> && requires(T t) {
    {std::complex<typename T::value_type>(t)};
};

template <class T>
concept FFT_Type = Complex<T> || std::floating_point<T>;
template <class Cont, class match_t>
concept MatchArray = std::ranges::contiguous_range<Cont> && std::same_as<
  match_t,
  typename std::remove_cvref_t<typename std::remove_reference_t<typename Cont::value_type>>>;

template <FFT_Type T>
using ComplexType = std::conditional_t<Complex<T>, T, std::complex<T>>;

template <FFT_Type time_t, Complex freq_t = ComplexType<time_t>>
class FFT {
   public:
    using complex_t = freq_t;
    using real_t = typename complex_t::value_type;

    // Okay to not be explicit here.  Any int is fine
    FFT(size_t fft_size, bool normalize = false) {  // NOLINT(runtime/explicit)
        _plan = std::shared_ptr<void>(detail::create_fft_plan<time_t>(),
                                      [](auto p) { detail::free_fft_plan<time_t>(p); });
        _fft_size = fft_size;
        _normalize = normalize;
    }

    void check_size(size_t length, std::string_view name) {
        if (length != _fft_size) {
            std::stringstream ss;
            ss << "len(" << name << ") = " << length << ", must be fftsize = " << _fft_size;   
            throw std::runtime_error(ss.str());
	}
    }

    void check_half_size(size_t length, std::string_view name) {
        if (length != (_fft_size / 2 + 1)) {
            std::stringstream ss;
            ss << "len(" << name << ") = " << length << ", must be fftsize/2+1 = " << _fft_size/2+1;   
            throw std::runtime_error(ss.str());
	}
    }

    template <MatchArray<complex_t> Out, MatchArray<complex_t> In>
    void execute_forward(Out& out, const In& in) {
        check_size(in.size(), "Input");
        check_size(out.size(), "Output");
        std::copy(in.begin(), in.end(), out.begin());
        detail::execute_fft(_plan.get(), out.data(), _fft_size, true, false);
    }

    template <MatchArray<complex_t> Out>
    void execute_forward(Out& out) {
        check_size(out.size(), "Output");
        detail::execute_fft(_plan.get(), out.data(), _fft_size, true, false);
    }

    template <MatchArray<complex_t> Out, MatchArray<real_t> In>
    void execute_forward(Out& out, const In& in) {
        check_size(in.size(), "Input");
        check_half_size(out.size(), "Output");
        real_scratch.resize(in.size());
        std::copy(in.begin(), in.end(), real_scratch.begin());
        detail::execute_r2c_fft(_plan.get(), out.data(), real_scratch.data(), _fft_size);
    }

    template <MatchArray<complex_t> Out, MatchArray<real_t> In>
    void execute_forward(Out& out, In& in) {
        check_size(in.size(), "Input");
        check_half_size(out.size(), "Output");
        detail::execute_r2c_fft(_plan.get(), out.data(), in.data(), _fft_size);
    }

    template <MatchArray<complex_t> Out, MatchArray<complex_t> In>
    void execute_inverse(Out& out, const In& in) {
        check_size(in.size(), "Input");
        check_size(out.size(), "Output");
        std::copy(in.begin(), in.end(), out.begin());
        detail::execute_fft(_plan.get(), out.data(), _fft_size, false, _normalize);
    }

    template <MatchArray<complex_t> Out>
    void execute_inverse(Out& out) {
        check_size(out.size(), "Output");
        detail::execute_fft(_plan.get(), out.data(), _fft_size, false, _normalize);
    }

    template <MatchArray<real_t> Out, MatchArray<complex_t> In>
    void execute_inverse(Out& out, const In& in) {
        check_half_size(in.size(), "Input");
        check_size(out.size(), "Output");
        complex_scratch.resize(in.size());
        std::copy(in.begin(), in.end(), complex_scratch.begin());
        detail::execute_c2r_fft(_plan.get(),
                                out.data(),
                                complex_scratch.data(),
                                _fft_size,
                                _normalize);
    }

    template <MatchArray<complex_t> Out, MatchArray<real_t> In>
    void execute_inverse(Out& out, In& in) {
        check_half_size(in.size(), "Input");
        check_size(out.size(), "Output");
        detail::execute_c2r_fft(_plan.get(), out.data(), in.data(), _fft_size, _normalize);
    }

   private:
    std::shared_ptr<void> _plan;
    size_t _fft_size;
    bool _normalize;
    std::vector<real_t> real_scratch;
    std::vector<complex_t> complex_scratch;
};

}  // namespace rustfft

#endif  // INCLUDE_FFTR_FFT_HPP_
