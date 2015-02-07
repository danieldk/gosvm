package gosvm

/*
#include <stdlib.h>
#include "wrap.h"
*/
import "C"

import (
	"runtime"
	"unsafe"
)

type mallocFunc func() unsafe.Pointer

// Mallocs, garbage-collects on fail, mallocs, panics on fail.
func tryNew(malloc mallocFunc) unsafe.Pointer {
	p := malloc()
	if p == nil {
		// Garbage-collect and try again.
		runtime.GC()
		p = malloc()
		if p == nil {
			panic("not enough memory")
		}
	}
	return p
}

func newLabels(n C.int) *C.int {
	labels := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.gosvm_labels_new(n))
	})
	return (*C.int)(labels)
}

func newProbs(model *C.svm_model_t) *C.double {
	probs := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.gosvm_probs_new(model))
	})
	return (*C.double)(probs)
}

func newDouble(n C.size_t) *C.double {
	p := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.gosvm_double_new(n))
	})
	return (*C.double)(p)
}

func newParameter() *C.svm_parameter_t {
	param := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.gosvm_parameter_new())
	})
	return (*C.svm_parameter_t)(param)
}

func newProblem() *C.svm_problem_t {
	problem := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.gosvm_problem_new())
	})
	return (*C.svm_problem_t)(problem)
}

func newNodes(n C.size_t) *C.svm_node_t {
	nodes := tryNew(func() unsafe.Pointer {
		return unsafe.Pointer(C.gosvm_nodes_new(n))
	})
	return (*C.svm_node_t)(nodes)
}
