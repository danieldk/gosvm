package gosvm

import "testing"

func TestFromDenseVector (t *testing.T) {
	fromDense := FromDenseVector([]float64{0.2, 0.1, 0.3, 0.6})
	sparse := FeatureVector{{1, 0.2}, {2, 0.1}, {3, 0.3}, {4, 0.6}}

	if (len(fromDense) != len(sparse)) {
		t.Errorf("len(fromDense) = %d, want %d", len(fromDense),
			len(sparse))
	}

	for idx, val := range sparse {
		if (fromDense[idx] != val) {
			t.Errorf("fromDense[%d] = (%d, %f), want (%d, %f)", idx,
				fromDense[idx].Index, fromDense[idx].Value, val.Index,
				val.Value)
		}
	}
}