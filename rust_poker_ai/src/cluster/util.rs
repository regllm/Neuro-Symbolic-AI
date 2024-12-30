use ckmeans::cluster_u8;
use simple_emd::distance;
use std::mem::drop;


const KMEANS_MAX_ITER: u32 = 300;


pub fn kmeans(
    data_points: &Vec<Vec<u8>>,
    cluster_count: u32,
) -> (Vec<Vec<f64>>, Vec<u32>) {
    let (centers, clusters) = cluster_u8(data_points, cluster_count, KMEANS_MAX_ITER);
    (centers, clusters)
}


pub fn wasserstein(v1: &Vec<f32>, v2: &Vec<f64>) -> f32 {
    // Ensure both vectors have the same length
    assert_eq!(v1.len(), v2.len(), "Input vectors must have the same length");

    let v2_f32: Vec<f32> = v2.iter().map(|&x| x as f32).collect();

    // let distance = distance(&v1, &v2_f32) as f32;

    // if (distance < 0.0) {
    //     println!("distance failed: {:?}, {:?}", v1, v2_f32);
    // }

    let mut result: f32;
    let mut count = 0;

    loop {
        result = distance(&v1, &v2_f32) as f32;
        if result >= 0.0 {
            break;
        }
        count += 1;
        println!("distance failed {:?}: {:?}, {:?}", count, v1, v2_f32);
    }

    drop(v2_f32);

    result
}

pub fn euclidean(v1: &Vec<f32>, v2: &Vec<f64>) -> f32 {
    assert_eq!(v1.len(), v2.len(), "Input vectors must have the same length");

    let mut sum = 0.0;
    for (a, b) in v1.iter().zip(v2.iter()) {
        let diff = f64::from(*a) - b;
        sum += diff.powi(2);
    }

    sum.sqrt() as f32
}
