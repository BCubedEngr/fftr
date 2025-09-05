use rustfft::num_complex::Complex;
use fftr::{create_fft_fc32, execute_fft_fc32, free_fft_fc32};

fn main() {
    let planner = create_fft_fc32();
    //let mut planner = FftPlanner::new();
    let n = 4096;
    //let mut scratch = vec![Complex{ re: 0.0f32, im:0.0f32 }; fft.get_inplace_scratch_len()];

    let mut output = vec![Complex{ re: 0.0f32, im:0.0f32 }; n];
    //let input = vec![Complex{ re: 0.0f32, im:0.0f32 }; n];
    for _n in 0..100000 {
        execute_fft_fc32(planner, output.as_mut_ptr(), n, true, false);
        execute_fft_fc32(planner, output.as_mut_ptr(), n, false, false);
        //process_fft_fc32_2(&mut planner, output.as_mut_ptr(), input.as_ptr(), n as i32);
        //let fft = planner.plan_fft_forward(n);
        //fft.process_with_scratch(&mut buffer, &mut scratch);
        //fft.process(&mut buffer);
    }
    free_fft_fc32(planner);
}
