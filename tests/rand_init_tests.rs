use linalgae_rs::core::matrix::Matrix;

// -------- Dimension Tests --------

#[test]
fn test_rand_init_dimensions_f64() {
    let m = Matrix::<f64>::rand_init(5, 7);
    assert_eq!(m.rows(), 5);
    assert_eq!(m.cols(), 7);
    assert_eq!(m.as_slice().len(), 35);
}

#[test]
fn test_rand_init_single_element_f64() {
    let m = Matrix::<f64>::rand_init(1, 1);
    assert_eq!(m.rows(), 1);
    assert_eq!(m.cols(), 1);
}

#[test]
fn test_rand_init_square_f64() {
    let m = Matrix::<f64>::rand_init(10, 10);
    assert_eq!(m.rows(), 10);
    assert_eq!(m.cols(), 10);
}

#[test]
fn test_rand_init_rectangular_f64() {
    let m = Matrix::<f64>::rand_init(3, 7);
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 7);
}

#[test]
fn test_rand_init_large_matrix_f64() {
    let m = Matrix::<f64>::rand_init(100, 100);
    assert_eq!(m.rows(), 100);
    assert_eq!(m.cols(), 100);
}

// -------- Value Tests --------

#[test]
fn test_rand_init_values_vary_f64() {
    let m1 = Matrix::<f64>::rand_init(3, 3);
    let m2 = Matrix::<f64>::rand_init(3, 3);
    // Very unlikely to be exactly equal with random values
    assert_ne!(m1.as_slice(), m2.as_slice());
}

#[test]
fn test_rand_init_not_all_zeros_f64() {
    let m = Matrix::<f64>::rand_init(10, 10);
    let sum = m.sum();
    // With 100 standard normal values, sum should not be exactly 0
    assert_ne!(sum, 0.0);
}

#[test]
fn test_rand_init_finite_values_f64() {
    let m = Matrix::<f64>::rand_init(5, 5);
    for val in m.as_slice() {
        assert!(val.is_finite());
    }
}

// -------- f32 Tests --------

#[test]
fn test_rand_init_dimensions_f32() {
    let m = Matrix::<f32>::rand_init(4, 4);
    assert_eq!(m.rows(), 4);
    assert_eq!(m.cols(), 4);
}

#[test]
fn test_rand_init_single_element_f32() {
    let m = Matrix::<f32>::rand_init(1, 1);
    assert_eq!(m.rows(), 1);
    assert_eq!(m.cols(), 1);
}

#[test]
fn test_rand_init_rectangular_f32() {
    let m = Matrix::<f32>::rand_init(2, 8);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 8);
}

#[test]
fn test_rand_init_values_vary_f32() {
    let m1 = Matrix::<f32>::rand_init(3, 3);
    let m2 = Matrix::<f32>::rand_init(3, 3);
    assert_ne!(m1.as_slice(), m2.as_slice());
}

#[test]
fn test_rand_init_not_all_zeros_f32() {
    let m = Matrix::<f32>::rand_init(10, 10);
    let sum = m.sum();
    assert_ne!(sum, 0.0);
}

#[test]
fn test_rand_init_finite_values_f32() {
    let m = Matrix::<f32>::rand_init(5, 5);
    for val in m.as_slice() {
        assert!(val.is_finite());
    }
}

// -------- Error Tests --------

#[test]
#[should_panic(expected = "Row dim should be positive!")]
fn test_rand_init_panic_zero_rows_f64() {
    let _ = Matrix::<f64>::rand_init(0, 5);
}

#[test]
#[should_panic(expected = "Col dim should be positive!")]
fn test_rand_init_panic_zero_cols_f64() {
    let _ = Matrix::<f64>::rand_init(5, 0);
}

#[test]
#[should_panic(expected = "Row dim should be positive!")]
fn test_rand_init_panic_zero_rows_f32() {
    let _ = Matrix::<f32>::rand_init(0, 5);
}

#[test]
#[should_panic(expected = "Col dim should be positive!")]
fn test_rand_init_panic_zero_cols_f32() {
    let _ = Matrix::<f32>::rand_init(5, 0);
}

#[test]
#[should_panic(expected = "Row dim should be positive!")]
fn test_rand_init_panic_both_zero_f64() {
    let _ = Matrix::<f64>::rand_init(0, 0);
}
