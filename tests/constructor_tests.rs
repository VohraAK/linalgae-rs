use linalgae_rs::core::matrix::Matrix;

// -------- new() Constructor Tests --------

#[test]
fn test_matrix_new_valid() {
    let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(m.is_ok());
    
    let m = m.unwrap();
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
}

#[test]
fn test_matrix_new_dimension_mismatch() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0]); // 3 elements for 2x2
    assert!(m.is_err());
}

// -------- full() Constructor Tests --------

#[test]
fn test_matrix_full() {
    let m = Matrix::full(2, 3, 7.5).unwrap();
    
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
    
    for val in m.as_slice() {
        assert_eq!(*val, 7.5);
    }
}

// -------- zeroes() Constructor Tests --------

#[test]
fn test_matrix_zeroes() {
    let m = Matrix::zeroes(3, 3).unwrap();
    
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 3);
    
    for val in m.as_slice() {
        assert_eq!(*val, 0.0);
    }
}

// -------- ones() Constructor Tests --------

#[test]
fn test_matrix_ones() {
    let m = Matrix::ones(3, 2).unwrap();
    
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 2);
    
    for val in m.as_slice() {
        assert_eq!(*val, 1.0);
    }
}

// -------- Accessor Tests --------

#[test]
fn test_matrix_as_slice() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let m = Matrix::new(2, 2, data.clone()).unwrap();
    
    assert_eq!(m.as_slice(), &data[..]);
}

#[test]
fn test_matrix_as_mut_slice() {
    let mut m = Matrix::zeroes(2, 2).unwrap();
    
    // modify via mutable slice
    m.as_mut_slice()[0] = 5.0;
    m.as_mut_slice()[3] = 10.0;
    
    assert_eq!(m.as_slice()[0], 5.0);
    assert_eq!(m.as_slice()[3], 10.0);
}

// -------- Display Test --------

#[test]
fn test_matrix_display() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let display = format!("{}", m);
    
    // just check it doesn't panic and contains expected elements
    assert!(display.contains("1.00"));
    assert!(display.contains("4.00"));
}

// -------- Edge Case Tests --------

#[test]
fn test_matrix_new_empty_vector() {
    let m = Matrix::new(2, 2, vec![]);
    assert!(m.is_err());
}

#[test]
fn test_matrix_new_too_many_elements() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(m.is_err());
}

#[test]
fn test_matrix_new_single_element() {
    let m = Matrix::new(1, 1, vec![42.0]).unwrap();
    assert_eq!(m.rows(), 1);
    assert_eq!(m.cols(), 1);
    assert_eq!(m.as_slice()[0], 42.0);
}

#[test]
fn test_matrix_new_with_negative_values() {
    let m = Matrix::new(2, 2, vec![-1.0, -2.0, -3.0, -4.0]).unwrap();
    assert_eq!(m.as_slice(), &[-1.0, -2.0, -3.0, -4.0]);
}

#[test]
fn test_matrix_new_with_infinity() {
    let m = Matrix::new(2, 2, vec![f64::INFINITY, f64::NEG_INFINITY, 1.0, 2.0]).unwrap();
    assert_eq!(m.as_slice()[0], f64::INFINITY);
    assert_eq!(m.as_slice()[1], f64::NEG_INFINITY);
}

#[test]
fn test_matrix_new_with_nan() {
    let m = Matrix::new(2, 2, vec![f64::NAN, 1.0, 2.0, 3.0]).unwrap();
    assert!(m.as_slice()[0].is_nan());
}

#[test]
fn test_matrix_full_negative() {
    let m = Matrix::full(2, 2, -5.0).unwrap();
    for val in m.as_slice() {
        assert_eq!(*val, -5.0);
    }
}

#[test]
fn test_matrix_full_infinity() {
    let m = Matrix::full(2, 2, f64::INFINITY).unwrap();
    for val in m.as_slice() {
        assert_eq!(*val, f64::INFINITY);
    }
}

#[test]
fn test_matrix_single_row() {
    let m = Matrix::new(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    assert_eq!(m.rows(), 1);
    assert_eq!(m.cols(), 5);
}

#[test]
fn test_matrix_single_column() {
    let m = Matrix::new(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    assert_eq!(m.rows(), 5);
    assert_eq!(m.cols(), 1);
}

#[test]
fn test_matrix_as_mut_slice_all_elements() {
    let mut m = Matrix::zeroes(3, 3).unwrap();
    
    for (i, val) in m.as_mut_slice().iter_mut().enumerate() {
        *val = i as f64;
    }
    
    for (i, val) in m.as_slice().iter().enumerate() {
        assert_eq!(*val, i as f64);
    }
}
