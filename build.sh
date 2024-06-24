for backend in {Eigen,Stl,Vdt,Mad_Transcendental_Fast,Mad_Transcendental_Faster,Mad_Transcendental_Fastest}; do
    cmake -S . -B build_${backend} -DCMAKE_BUILD_TYPE=Release -DOPERON_MATH_BACKEND=${backend}
    cmake --build build_${backend} -j
done
