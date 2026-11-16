"""20_pca_ridge: ported from the teammate branch (its REPORT.md section 12) through code/classic_adapter.py."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
from classic_adapter import harness_predict

predict = harness_predict('pca_ridge')


if __name__ == '__main__':
    main(predict, __file__)
