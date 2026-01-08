use linalgae_rs::core::matrix::Matrix;

// -------- Constructor Tests --------

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

#[test]
fn test_matrix_zeroes() {
    let m = Matrix::zeroes(3, 3).unwrap();
    
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 3);
    
    // all elements should be zero
    for val in m.as_slice() {
        assert_eq!(*val, 0.0);
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

// -------- Arithmetic Tests --------

#[test]
fn test_matrix_addition() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    
    let c = &a + &b;
    
    assert_eq!(c.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_matrix_subtraction() {
    let a = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let c = &a - &b;
    
    assert_eq!(c.as_slice(), &[4.0, 4.0, 4.0, 4.0]);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_matrix_addition_dimension_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    let _ = &a + &b; // should panic
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_matrix_subtraction_dimension_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    let _ = &a - &b; // should panic
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
