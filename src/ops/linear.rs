use std::ops;
use crate::core::matrix::Matrix;
use num_traits::Num;


//----------Linear Algebra Operations----------//

// inner product / matmul
impl<T> ops::Mul<&Matrix<T>> for &Matrix<T>
where T: Num + Copy + ops::AddAssign
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output 
    {
        // let lhs_cols = self.cols();
        // let rhs_rows = rhs.rows();
        
        // // check dims
        // if lhs_cols != rhs_rows
        // {
        //     panic!("Matrix::Mul: lhs_cols ({}) != rhs_rows ({})", lhs_cols, rhs_rows);
        // }
        
        // let lhs_rows = self.rows();
        // let rhs_cols = rhs.cols();

        // let lhs_slice = self.as_slice();
        // let rhs_slice = rhs.as_slice();

        // let mut result = vec![T::zero(); lhs_rows * rhs_cols];

        // // optimised version of matmul, should be much faster
        // for i in 0..lhs_rows 
        // {
        //     for k in 0..lhs_cols 
        //     {
        //         let lhs_val = lhs_slice[i * lhs_cols + k];
                
        //         for j in 0..rhs_cols 
        //         {
        //             let rhs_val = rhs_slice[k * rhs_cols + j];
        //             let res_idx = i * rhs_cols + j;
        //             result[res_idx] += lhs_val * rhs_val;
        //         }
        //     }
        // }

        // Matrix::new(lhs_rows, rhs_cols, result).expect("Matrix::Mul: Result has incorrect dimensions!")

        self.matmul(rhs)
    }
}