use rustfft::{FftPlanner, num_complex::Complex};
use realfft::RealFftPlanner;
use std::slice;

#[no_mangle]
pub extern "C" fn execute_fft_fc32(planner: *mut FftPlanner<f32>, output: *mut Complex<f32>, fft_size: usize, forward: bool, normalize: bool) {
    let plan = unsafe{&mut *planner};
    // FFTs are in place
    let mut output_slice = unsafe{slice::from_raw_parts_mut(output, fft_size)};
    if forward {
        plan.plan_fft_forward(fft_size).process(&mut output_slice);
    } else {
        plan.plan_fft_inverse(fft_size).process(&mut output_slice);
        if normalize {
            let scale = 1.0 / fft_size as f32;
            for i in output_slice.iter_mut() {
                *i *= scale;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn execute_fft_fc64(planner: *mut FftPlanner<f64>, output: *mut Complex<f64>, fft_size: usize, forward: bool, normalize: bool) {
    let plan = unsafe{&mut *planner};
    // FFTs are in place
    let mut output_slice = unsafe{slice::from_raw_parts_mut(output, fft_size)};
    if forward {
        plan.plan_fft_forward(fft_size).process(&mut output_slice);
    } else {
        plan.plan_fft_inverse(fft_size).process(&mut output_slice);
        if normalize {
            let scale = 1.0 / fft_size as f64;
            for i in output_slice.iter_mut() {
                *i *= scale;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn create_fft_fc32() -> *mut FftPlanner<f32> {
    return Box::into_raw(Box::new(FftPlanner::new()));
    // let planner: FftPlanner<f32> = FftPlanner::new();
    // let arc_planner = Arc::new(planner);
    // return Arc::into_raw(arc_planner) as *mut FftPlanner<f32>;
}

#[no_mangle]
pub extern "C" fn create_fft_fc64() -> *mut FftPlanner<f64> {
    return Box::into_raw(Box::new(FftPlanner::new()));
    // let planner: FftPlanner<f64> = FftPlanner::new();
    // let arc_planner = Arc::new(planner);
    // return Arc::into_raw(arc_planner) as *mut FftPlanner<f64>;
}

#[no_mangle]
pub extern "C" fn free_fft_fc32(plan: *mut FftPlanner<f32>) {
    unsafe{drop(Box::from_raw(plan))};
}

#[no_mangle]
pub extern "C" fn free_fft_fc64(plan: *mut FftPlanner<f64>) {
    unsafe{drop(Box::from_raw(plan))};
}

#[no_mangle]
pub extern "C" fn execute_r2c_fft_fc32(planner: *mut RealFftPlanner<f32>, output: *mut Complex<f32>, input: *mut f32, fft_size: usize) {
    let plan = unsafe{&mut *planner};
    let mut input_slice = unsafe{slice::from_raw_parts_mut(input, fft_size)};
    let mut output_slice = unsafe{slice::from_raw_parts_mut(output, fft_size/2 + 1)};
    plan.plan_fft_forward(fft_size).process(&mut input_slice, &mut output_slice).unwrap();
}

#[no_mangle]
pub extern "C" fn execute_r2c_fft_fc64(planner: *mut RealFftPlanner<f64>, output: *mut Complex<f64>, input: *mut f64, fft_size: usize) {
    let plan = unsafe{&mut *planner};
    let mut input_slice = unsafe{slice::from_raw_parts_mut(input, fft_size)};
    let mut output_slice = unsafe{slice::from_raw_parts_mut(output, fft_size/2 + 1)};
    plan.plan_fft_forward(fft_size).process(&mut input_slice, &mut output_slice).unwrap();
}

#[no_mangle]
pub extern "C" fn execute_c2r_fft_fc32(planner: *mut RealFftPlanner<f32>, output: *mut f32, input: *mut Complex<f32>, fft_size: usize, normalize: bool) {
    let plan = unsafe{&mut *planner};
    let mut input_slice = unsafe{slice::from_raw_parts_mut(input, fft_size/2 + 1)};
    let mut output_slice = unsafe{slice::from_raw_parts_mut(output, fft_size)};
    plan.plan_fft_inverse(fft_size).process(&mut input_slice, &mut output_slice).unwrap();
    if normalize {
        let scale = 1.0 / fft_size as f32;
        for i in output_slice.iter_mut() {
            *i *= scale;
        }
    }
}

#[no_mangle]
pub extern "C" fn execute_c2r_fft_fc64(planner: *mut RealFftPlanner<f64>, output: *mut f64, input: *mut Complex<f64>, fft_size: usize, normalize: bool) {
    let plan = unsafe{&mut *planner};
    let mut input_slice = unsafe{slice::from_raw_parts_mut(input, fft_size/2 + 1)};
    let mut output_slice = unsafe{slice::from_raw_parts_mut(output, fft_size)};
    plan.plan_fft_inverse(fft_size).process(&mut input_slice, &mut output_slice).unwrap();
    if normalize {
        let scale = 1.0 / fft_size as f64;
        for i in output_slice.iter_mut() {
            *i *= scale;
        }
    }
}

#[no_mangle]
pub extern "C" fn create_r2c_fft_fc32() -> *mut RealFftPlanner<f32> {
    return Box::into_raw(Box::new(RealFftPlanner::new()));
}

#[no_mangle]
pub extern "C" fn create_r2c_fft_fc64() -> *mut RealFftPlanner<f64> {
    return Box::into_raw(Box::new(RealFftPlanner::new()));
}

#[no_mangle]
pub extern "C" fn free_r2c_fft_fc32(plan: *mut RealFftPlanner<f32>) {
    unsafe{drop(Box::from_raw(plan))};
}

#[no_mangle]
pub extern "C" fn free_r2c_fft_fc64(plan: *mut RealFftPlanner<f64>) {
    unsafe{drop(Box::from_raw(plan))};
}
