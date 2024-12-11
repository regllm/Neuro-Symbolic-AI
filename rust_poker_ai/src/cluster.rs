// use linfa::{DatasetBase, DatasetView};
// use linfa::prelude::Predict;
// use linfa::traits::Fit;
// use linfa_clustering::KMeans;
// use ndarray::{Array2, ArrayBase, OwnedRepr, Dim, Ix1};
use linfa::DatasetBase;
use linfa::traits::{Fit, FitWith, Predict};
use linfa_clustering::{KMeansParams, KMeans, IncrKMeansError};
use linfa_datasets::generate;
use ndarray::{Axis, array, s};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use approx::assert_abs_diff_eq;

const MAX_ITER: u64 = 300;


pub fn kmeans(
    data_points: &Vec<Vec<f32>>,
    cluster_count: u16,
) {
    // let array_data = Array2::from_shape_fn((data_points.len(), data_points[0].len()), |(i, j)| {
    //     data_points[i][j]
    // });

    // // let dataset: DatasetBase<ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>, ()> = DatasetBase::new(data_array, ());
    // // let dataset: DatasetView<f64, usize, Ix1> = (data_array.view(), ()).into();
    // let dataset = DatasetBase::from(array_data);

    // let kmeans = KMeans::params(cluster_count as usize).fit(&dataset).unwrap();
    // let labels = kmeans.predict(&dataset).into_raw_vec();

    // println!("{:?}", labels);
    
    // Our random number generator, seeded for reproducibility
    let seed = 42;
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);

    // `expected_centroids` has shape `(n_centroids, n_features)`
    // i.e. three points in the 2-dimensional plane
    let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
    // Let's generate a synthetic dataset: three blobs of observations
    // (100 points each) centered around our `expected_centroids`
    let data = generate::blobs(100, &expected_centroids, &mut rng);
    let n_clusters = expected_centroids.len_of(Axis(0));

    // Standard K-means
    {
        let observations = DatasetBase::from(data.clone());
        // Let's configure and run our K-means algorithm
        // We use the builder pattern to specify the hyperparameters
        // `n_clusters` is the only mandatory parameter.
        // If you don't specify the others (e.g. `n_runs`, `tolerance`, `max_n_iterations`)
        // default values will be used.
        let model = KMeans::params_with_rng(n_clusters, rng.clone())
            .tolerance(1e-2)
            .fit(&observations)
            .expect("KMeans fitted");

        // Once we found our set of centroids, we can also assign new points to the nearest cluster
        let new_observation = DatasetBase::from(array![[-9., 20.5]]);
        // Predict returns the **index** of the nearest cluster
        let dataset = model.predict(new_observation);
        // We can retrieve the actual centroid of the closest cluster using `.centroids()`
        let closest_centroid = &model.centroids().index_axis(Axis(0), dataset.targets()[0]);
        assert_abs_diff_eq!(closest_centroid.to_owned(), &array![-10., 20.], epsilon = 1e-1);

        println!("Result: {:?}", closest_centroid);
    }
}
