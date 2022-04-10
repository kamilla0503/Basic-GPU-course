#include <iomanip>
#include <iostream>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>

/*
template<typename Number>
//void element(const Number* J, Number*  A)
void element(Kokkos::View<double [3][3]> J, Kokkos::View<double [4][4]> A)
{
    Number C0 = J(1, 1) * J(2, 2) - J(1, 2) * J(2, 1);
    Number C1 = J(1, 2) * J(2, 0) - J(1, 0) * J(2, 2);
    Number C2 = J(1, 0) * J(2, 1) - J(1, 1) * J(2, 0);
    Number inv_J_det = J(0, 0) * C0 + J(0, 1) * C1 + J(0, 2) * C2;
    Number d = (1./6.) / inv_J_det;
    Number G0 = d * (J(0,0) * J(0,0) + J(1,0) * J(1,0) + J(2,0) * J(2,0));
    Number G1 = d * (J(0,0) * J(0,1) + J(1,0) * J(1,1) + J(2,0) * J(2,1));
    Number G2 = d * (J(0,0) * J(0,2) + J(1,0) * J(1,2) + J(2,0) * J(2,2));
    Number G3 = d * (J(0,1) * J(0,1) + J(1,1) * J(1,1) + J(2,1) * J(2,1));
    Number G4 = d * (J(0,1) * J(0,2) + J(1,1) * J(1,2) + J(2,1) * J(2,2));
    Number G5 = d * (J(0,2) * J(0,2) + J(1,2) * J(1,2) + J(2,2) * J(2,2));

    A(0, 0) = G0;
    A(0, 1) = A(1, 0) = G1;
    A(0, 2) = A(2, 0) = G2;
    A(0, 3) = A(3, 0) = -G0 - G1 - G2;
    A(1, 1) = G3;
    A(1, 2) = A(2, 1) = G4;
    A(1, 3) = A(3, 1) = -G1 - G3 - G4;
    A(2, 2) = G5;
    A(2, 3) = A(3, 2) = -G2 - G4 - G5;
    A(3, 3) = G0 + 2 * G1 + 2 * G2 + G3 + 2 * G4 + G5;

}
*/


int main( int argc, char* argv[] ) {
    int Nmin = 1000;
    int Nmax = 40000000;
    int nrepeat = 20;

    Kokkos::initialize(argc, argv);
    {

#ifdef KOKKOS_ENABLE_OPENMP
        #define OMP_PROC_BIND spread
#define OMP_PLACES threads
#endif
#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#endif
#ifdef KOKKOS_ENABLE_HIP
#define MemSpace Kokkos::Experimental::HIPSpace
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
#define MemSpace Kokkos::OpenMPTargetSpace
#endif
#ifdef KOKKOS_RIGHT
#define LayoutType Kokkos::LayoutLeft
#endif


#ifndef MemSpace
#define MemSpace Kokkos::HostSpace
#endif
#ifndef LayoutType
#define LayoutType Kokkos::LayoutRight
#endif
    for (int N = Nmin; N<=Nmax; N+=100){





//1 run-time, 2 compile-time
        Kokkos::View<double *[3][3], LayoutType, MemSpace> J("J", N);
        Kokkos::View<double *[4][4], LayoutType, MemSpace> A("A", N);

        Kokkos::View<double *[3][3], LayoutType, MemSpace>::HostMirror h_J = Kokkos::create_mirror_view(J);
        Kokkos::View<double *[4][4], LayoutType, MemSpace>::HostMirror h_A = Kokkos::create_mirror_view(A);


// Initialize inverse Jacobian matrices on host.
        for (int i = 0; i < N; ++i) {
            h_J(i, 0, 0) = 3;
            h_J(i, 0, 1) = 1;
            h_J(i, 0, 2) = 1;
            h_J(i, 1, 0) = 1;
            h_J(i, 1, 2) = 1;
            h_J(i, 1, 1) = 3;
            h_J(i, 2, 2) = 3;
            h_J(i, 2, 0) = 1;
            h_J(i, 2, 1) = 1;
            //h_J(i) = {3,1,1,1,3,1,1,1,3};
            /*{{3., 1., 1.},
                      {1., 3., 1.},
                      {1., 1., 3.}};*/
        }

// Initialize A on host.
        for (int i = 0; i < N; ++i) {
            for (int m = 0; m < 4; m++) {
                for (int n = 0; n < 4; n++)
                    h_A(i, m, n) = 0;
            }
        }

        Kokkos::deep_copy(J, h_J);
        Kokkos::deep_copy(A, h_A);

// Timer products.
        Kokkos::Timer timer;

        //double avg t = 0.0;

        for (int repeat = 0; repeat < nrepeat; repeat++) {
            //compute element matrices A_ij
            //Kokkos::parallel_for ('My_kernel', N, [=] (const int64_t i)
            //Kokkos::parallel_for ('My_kernel', N,  KOKKOS_LAMBDA(const int64_t i)
            Kokkos::parallel_for(N, KOKKOS_LAMBDA(
            const int64_t i)
            {
                auto C0 = J(i, 1, 1) * J(i, 2, 2) - J(i, 1, 2) * J(i, 2, 1);
                auto C1 = J(i, 1, 2) * J(i, 2, 0) - J(i, 1, 0) * J(i, 2, 2);
                auto C2 = J(i, 1, 0) * J(i, 2, 1) - J(i, 1, 1) * J(i, 2, 0);
                auto inv_J_det = J(i, 0, 0) * C0 + J(i, 0, 1) * C1 + J(i, 0, 2) * C2;
                auto d = (1. / 6.) / inv_J_det;
                auto G0 = d * (J(i, 0, 0) * J(i, 0, 0) + J(i, 1, 0) * J(i, 1, 0) + J(i, 2, 0) * J(i, 2, 0));
                auto G1 = d * (J(i, 0, 0) * J(i, 0, 1) + J(i, 1, 0) * J(i, 1, 1) + J(i, 2, 0) * J(i, 2, 1));
                auto G2 = d * (J(i, 0, 0) * J(i, 0, 2) + J(i, 1, 0) * J(i, 1, 2) + J(i, 2, 0) * J(i, 2, 2));
                auto G3 = d * (J(i, 0, 1) * J(i, 0, 1) + J(i, 1, 1) * J(i, 1, 1) + J(i, 2, 1) * J(i, 2, 1));
                auto G4 = d * (J(i, 0, 1) * J(i, 0, 2) + J(i, 1, 1) * J(i, 1, 2) + J(i, 2, 1) * J(i, 2, 2));
                auto G5 = d * (J(i, 0, 2) * J(i, 0, 2) + J(i, 1, 2) * J(i, 1, 2) + J(i, 2, 2) * J(i, 2, 2));

                A(i, 0, 0) = G0;
                A(i, 0, 1) = A(i, 1, 0) = G1;
                A(i, 0, 2) = A(i, 2, 0) = G2;
                A(i, 0, 3) = A(i, 3, 0) = -G0 - G1 - G2;
                A(i, 1, 1) = G3;
                A(i, 1, 2) = A(i, 2, 1) = G4;
                A(i, 1, 3) = A(i, 3, 1) = -G1 - G3 - G4;
                A(i, 2, 2) = G5;
                A(i, 2, 3) = A(i, 3, 2) = -G2 - G4 - G5;
                A(i, 3, 3) = G0 + 2 * G1 + 2 * G2 + G3 + 2 * G4 + G5;

                //element( Kokkos::subview (J, i,  Kokkos::ALL, Kokkos::ALL),
                //        Kokkos::subview (A, i, Kokkos::ALL, Kokkos::ALL));
                //element(J[i], A[i]);
            }  );


        }

            Kokkos::fence();

            double time = timer.seconds();

        // Calculate bandwidth.
        double Gbytes = 1.0e-9 * double(sizeof(double) * (21 * N));

        std::cout << "size " << std::setw(8) << N << " "
                  << 1e-6 * N / time / nrepeat
                  << " MUPD/s or " << std::setw(8)
                  << Gbytes / time / nrepeat << " GB/s" << std::endl;

        //for checking
        /**for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                std::cout << A(50, i, j) << " ";
            }
            std::cout << std::endl;
        }**/

    }
    Kokkos::finalize();


}


}