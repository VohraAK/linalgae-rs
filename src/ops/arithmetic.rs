use std::ops;
use crate::core::matrix::Matrix;
use num_traits::Num;

// ----------Arithmetic Operations----------//

// addition
impl<T> ops::Add<&Matrix<T>> for &Matrix<T>
where T: Num + Copy + ops::Add<Output = T>
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output 
    {
        let lhs_rows = self.rows();
        let lhs_cols = self.cols();
        let rhs_rows = rhs.rows();
        let rhs_cols = rhs.cols();

        // ensure that the dims match up
        if lhs_rows != rhs_rows || lhs_cols != rhs_cols
        {
            panic!("Matrix::Add: Dimension mismatch! ( ({}, {}) vs ({}, {}) )", lhs_rows, lhs_cols, rhs_rows, rhs_cols);
        }

        let result = self.as_slice().iter().zip(rhs.as_slice().iter()).map(|(a, b)| *a + *b ).collect();

        Matrix::new(lhs_rows, lhs_cols, result).expect("Matrix::Add: Result has incorrect dimensions!")
        
    }
}

// subtraction
impl<T> ops::Sub<&Matrix<T>> for &Matrix<T>
where T: Num + Copy + ops::Sub<Output = T>
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output 
    {
        let lhs_rows = self.rows();
        let lhs_cols = self.cols();
        let rhs_rows = rhs.rows();
        let rhs_cols = rhs.cols();

        // ensure that the dims match up
        if lhs_rows != rhs_rows || lhs_cols != rhs_cols
        {
            panic!("Matrix::Sub: Dimension mismatch! ( ({}, {}) vs ({}, {}) )", lhs_rows, lhs_cols, rhs_rows, rhs_cols);
        }

        let result = self.as_slice().iter().zip(rhs.as_slice().iter()).map(|(a, b)| *a - *b ).collect();

        Matrix::new(lhs_rows, lhs_cols, result).expect("Matrix::Sub: Result has incorrect dimensions!")
        
    }
}

// scalar mul (Matrix * scalar)
impl<T, U> ops::Mul<U> for &Matrix<T>
where T: Num + Copy + ops::Mul<U, Output = T>, U: Num + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: U) -> Self::Output 
    {
        let lhs_rows = self.rows();
        let lhs_cols = self.cols();

        let result = self.as_slice().iter().map(|a| *a * rhs).collect();

        Matrix::new(lhs_rows, lhs_cols, result).expect("Matrix::Mul: Result has incorrect dimensions!")
    }
}

// NOTE: removed this scalar mul fn, due to orphan-rule violation (could fix this but it would be a pain in the arse)
// scalar mul (scalar * Matrix)
// impl<T: Num, U: Num> ops::Mul<&Matrix<T>> for U
// where T: Copy + ops::Mul<U, Output = T>, U: Copy,
// {
//     type Output = Matrix<T>;

//     fn mul(self, rhs: &Matrix<T>) -> Self::Output 
//     {
//         rhs * self
//     }
// }

// scalar div (Matrix / scalar)
impl<T, U> ops::Div<U> for &Matrix<T>
where T: Num + Copy + ops::Div<U, Output = T>, U: Num + Copy
{
    type Output = Matrix<T>;

    fn div(self, rhs: U) -> Self::Output 
    {
        // TODO: epsilon
        // check 0-divisor
        if rhs == (U::zero())
        {
            panic!("Matrix::Div: Cannot divide by zero!");
        }

        let lhs_rows = self.rows();
        let lhs_cols = self.cols();

        let result = self.as_slice().iter().map(|a| *a / rhs).collect();

        Matrix::new(lhs_rows, lhs_cols, result).expect("Matrix::Div: Result has incorrect dimensions!") 
    }
}

// inplace add_assign
impl<T> ops::AddAssign<&Matrix<T>> for Matrix<T>
where T: Num + Copy + ops::AddAssign,
{
    fn add_assign(&mut self, rhs: &Matrix<T>) 
    {
        let lhs_rows = self.rows();
        let lhs_cols = self.cols();
        let rhs_rows = rhs.rows();
        let rhs_cols = rhs.cols();

        if lhs_rows != rhs_rows || lhs_cols != rhs_cols 
        {
            panic!("Matrix::AddAssign: Dimension mismatch!");
        }

        for (l, r) in self.as_mut_slice().iter_mut().zip(rhs.as_slice().iter()) 
        {
            *l += *r;
        }
    }   
}

// inplace sub_assign
impl<T> ops::SubAssign<&Matrix<T>> for Matrix<T>
where T: Num + Copy + ops::SubAssign,
{
    fn sub_assign(&mut self, rhs: &Matrix<T>) 
    {    
        let lhs_rows = self.rows();
        let lhs_cols = self.cols();
        let rhs_rows = rhs.rows();
        let rhs_cols = rhs.cols();

        if lhs_rows != rhs_rows || lhs_cols != rhs_cols 
        {
            panic!("Matrix::SubAssign: Dimension mismatch!");
        }

        for (l, r) in self.as_mut_slice().iter_mut().zip(rhs.as_slice().iter()) 
        {
            *l -= *r;
        }
    }   
}
