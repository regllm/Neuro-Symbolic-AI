use simple_emd::distance;
// use ndarray::Array1;
// use num_traits::{Num, ToPrimitive};
use std::mem::drop;

// pub fn wasserstein<T: Num + ToPrimitive, T2: Num + ToPrimitive>(v1: &Vec<T>, v2: &Vec<T2>) -> f64 {
//     // Ensure both vectors have the same length
//     assert_eq!(v1.len(), v2.len(), "Input vectors must have the same length");

//     vectors_distance(&arr1.view(), &arr2.view()) as f64
// }

pub fn wasserstein(v1: &Vec<f32>, v2: &Vec<f64>) -> f32 {
    // Ensure both vectors have the same length
    assert_eq!(v1.len(), v2.len(), "Input vectors must have the same length");

    let v2_f32: Vec<f32> = v2.iter().map(|&x| x as f32).collect();

    let distance = distance(&v1, &v2_f32) as f32;

    drop(v2_f32);

    distance
}
