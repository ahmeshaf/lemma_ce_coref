import pickle

ecb_mention_map = pickle.load(open('mention_map.pkl', 'rb'))

doc2mentions = {}

for m in ecb_mention_map.values():
    m['predicted_topic'] = None
    if m['doc_id'] not in doc2mentions:
        doc2mentions[m['doc_id']] = []
    doc2mentions[m['doc_id']].append(m)

predicted_tops = pickle.load(open('predicted_topics', 'rb'))

for i, docs in enumerate(predicted_tops):
    pred_top = f"predicted_{i}"
    for doc in docs:
        doc_xml = doc + ".xml"
        if doc_xml in doc2mentions:
            for m in doc2mentions[doc_xml]:
                m['predicted_topic'] = pred_top

pickle.dump(ecb_mention_map, open('mention_map.pkl', 'wb'))

pass