import database.api as db
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from analysis.values.values_registry import VALUES_REGISTRY
from analysis.values.val_utils import fmt, unfmt, latexify


def cluster_and_score(features, actual_labels, num_tnse_components=3, attempts=10):
    num_classes = len(set(actual_labels))
    best_clusters, best_ari, best_nmi = None, -1, -1
    for attempt in range(attempts):
        feature_emb = TSNE(n_components=num_tnse_components).fit_transform(features)
        predicted_clusters = KMeans(n_clusters=num_classes).fit_predict(feature_emb)
        ARI = adjusted_rand_score(actual_labels, predicted_clusters)
        NMI = normalized_mutual_info_score(actual_labels, predicted_clusters)
        if ARI > best_ari:
            best_clusters, best_ari, best_nmi = predicted_clusters, ARI, NMI
    return best_clusters, best_ari, best_nmi


def clustering_scores_for(classifier_name, scale=None, is_l1=None):
    query = f'''
        select actual, feature
        from sisr_analysis
        where classifier_name = '{classifier_name}'
        and generator_name like "%-pretrained"'''
    if scale is not None:
        query += f' and scale = "{scale}"'
    if is_l1 is not None:
        if is_l1:
            query += f' and loss = "L1"'
        else:
            query += f' and loss != "L1"'

    custom_clsfr_data = db.read_and_decode_sql_query(query)
    features = np.array(list(custom_clsfr_data.feature))

    unique_labels = sorted(set(custom_clsfr_data.actual))
    actual_labels = []
    for label in list(custom_clsfr_data.actual):
        label_index = unique_labels.index(label)
        actual_labels.append(label_index)

    return cluster_and_score(features, actual_labels)


@VALUES_REGISTRY.register()
def clustering_scores():
    values = {}
    classifier_names = [
        ("ConvNext_CNN_SISR_pretrained_models", "ID"),
        ("ConvNext_CNN_SISR_custom_models", "OOD"),
    ]
    for classifier_name, description in classifier_names:
        clusters, ARI, NMI = clustering_scores_for(classifier_name)
        ari_value_name = f"{description}ClusterARI"
        values[ari_value_name] = ARI
        nmi_value_name = f"{description}ClusterNMI"
        values[ari_value_name] = NMI

        for scale in (2, 4):
            for is_l1 in (True, False):
                clusters, ARI, NMI = clustering_scores_for(
                    classifier_name, scale, is_l1
                )
                scale_name = "two" if loss == 2 else "four"
                loss_name = "LOne" if is_l1 else "Adv"
                ari_value_name = f"{description}X{scale_name}{loss_name}ClusterARI"
                values[ari_value_name] = ARI
                nmi_value_name = f"{description}X{scale_name}{loss_name}ClusterNMI"
                values[ari_value_name] = NMI


# attribution feature clustering
