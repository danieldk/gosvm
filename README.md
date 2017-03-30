## Warning

I put this repository back after someone's request. I do not intend to
maintain it or fix any issues with it ;).

## Introduction

gosvm is a package for training and using support vector machines (SVM)
in the Go programming language.

**Note:** This package is still new, its API may change, and not all
of libsvm's functionality may be available yet.

## Installation

The <tt>go</tt> command can be used to install this package:

    go get github.com/danieldk/gosvm

The package documentation is available at: http://go.pkgdoc.org/github.com/danieldk/gosvm

## OpenMP

If you wish to use <tt>libsvm</tt> with OpenMP support for multicore processing, please use this command to install the package:

    CGO_LDFLAGS="-lgomp" go get github.com/danieldk/gosvm

## Examples

Examples for using gosvm can be found at:

https://github.com/danieldk/gosvm-examples

## Links

There is a port of libsvm to Go (rather than a binding) at:

https://github.com/ewalker544/libsvm-go
