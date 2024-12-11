use emd::distance;
use ndarray::Array1;


pub fn wasserstein(v1: &Vec<f32>, v2: &Vec<f32>) -> f32 {
    // Ensure both vectors have the same length
    assert_eq!(v1.len(), v2.len(), "Input vectors must have the same length");

    let arr1: Array1<f64> = v1.iter().map(|&x| x.into()).collect::<Array1<f64>>();
    let arr2: Array1<f64> = v2.iter().map(|&x| x.into()).collect::<Array1<f64>>();
    distance(&arr1.view(), &arr2.view()) as f32
}
