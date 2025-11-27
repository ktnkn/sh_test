from setuptools import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

cppshogi_dir = r'E:\python\DeepLearningShogi\cppshogi'

extensions = [
    Extension(
        "cppshogi",
        [
            r"E:\python\DeepLearningShogi\dlshogi\cppshogi.pyx",
            f"{cppshogi_dir}\\bitboard.cpp",
            f"{cppshogi_dir}\\book.cpp",
            f"{cppshogi_dir}\\common.cpp",
            f"{cppshogi_dir}\\cppshogi.cpp",
            f"{cppshogi_dir}\\dtype.cpp",
            f"{cppshogi_dir}\\generateMoves.cpp",
            f"{cppshogi_dir}\\hand.cpp",
            f"{cppshogi_dir}\\init.cpp",
            f"{cppshogi_dir}\\move.cpp",
            f"{cppshogi_dir}\\mt64bit.cpp",
            f"{cppshogi_dir}\\position.cpp",
            f"{cppshogi_dir}\\python_module.cpp",
            f"{cppshogi_dir}\\search.cpp",
            f"{cppshogi_dir}\\square.cpp",
            f"{cppshogi_dir}\\usi.cpp",
        ],
        include_dirs=['.', get_include(), cppshogi_dir],
        language="c++",
        extra_compile_args=['/std:c++17'],
    )
]

setup(
    name="dlshogi",
    ext_modules=cythonize(extensions)
)