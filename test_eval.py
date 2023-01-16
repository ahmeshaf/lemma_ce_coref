from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import evaluate_documents as evaluate
from coval.coval.eval.evaluator import muc, b_cubed, ceafe, lea


def read(key, response):
    return get_coref_infos('%s' % key, '%s' % response,
            False, False, True)


doc = read('../coreference/test_evt_gold.keyfile', '../coreference/evt_system.keyfile')

print(evaluate(doc, ceafe))
