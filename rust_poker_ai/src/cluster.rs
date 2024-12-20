use ckmeans::cluster_u8;
const MAX_ITER: u32 = 300;


pub fn kmeans(
    data_points: &Vec<Vec<u8>>,
    cluster_count: u32,
) -> (Vec<Vec<f64>>, Vec<u32>) {
    let (centers, clusters) = cluster_u8(data_points, cluster_count, MAX_ITER);
    (centers, clusters)
}
