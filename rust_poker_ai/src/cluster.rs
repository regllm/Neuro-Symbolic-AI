use ckmeans::cluster_with_kmeans;
const MAX_ITER: u32 = 300;


pub fn kmeans(
    data_points: &Vec<Vec<f64>>,
    cluster_count: u32,
) -> (Vec<Vec<f64>>, Vec<u32>) {
    let (centers, clusters) = cluster_with_kmeans(data_points, cluster_count, MAX_ITER);
    (centers, clusters)
}
