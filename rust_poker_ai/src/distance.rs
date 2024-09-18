use emd::distance;
use ndarray::Array1;
use num_traits::{Num, ToPrimitive};


pub fn wasserstein<T: Num + ToPrimitive, T2: Num + ToPrimitive>(v1: &Vec<T>, v2: &Vec<T2>) -> f64 {
    // Ensure both vectors have the same length
    assert_eq!(v1.len(), v2.len(), "Input vectors must have the same length");

    let arr1: Array1<f64> = v1.iter().map(|x| x.to_f64().unwrap()).collect::<Array1<f64>>();
    let arr2: Array1<f64> = v2.iter().map(|x| x.to_f64().unwrap()).collect::<Array1<f64>>();
    distance(&arr1.view(), &arr2.view()) as f64
}
