"""16_embedding: ported from the teammate branch (its methods/07_embedding) through code/emb_adapter.py."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import main
from emb_adapter import harness_predict

predict = harness_predict('embed')


if __name__ == '__main__':
    main(predict, __file__)
