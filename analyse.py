"""Reproduce an exploratory workplace-support segmentation of OSMI 2016.

Run from any directory: python analyse.py. No individual diagnostic inference.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import pdist, squareform
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
FIG = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
SEED = 42

def hierarchical(distance, k):
    return AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average').fit_predict(distance)

def main():
    raw_path = ROOT / 'data/survey_2016.csv'
    raw = pd.read_csv(raw_path)
    cols = raw.columns
    audit = pd.DataFrame({'index': range(len(cols)), 'question': cols,
                          'missing_n': raw.isna().sum().values,
                          'missing_pct': raw.isna().mean().values * 100,
                          'unique_nonmissing': raw.nunique().values})
    audit.to_csv(OUT / 'data_audit.csv', index=False)
    # Question 3 is only asked of respondents outside primarily tech companies.
    mask = raw.iloc[:, 0].eq(0) & (raw.iloc[:, 2].eq(1) | raw.iloc[:, 3].eq(1))
    population = raw.loc[mask].copy()
    feature_indices = list(range(4, 16))
    names = ['benefits', 'care_options', 'formal_discussion', 'help_resources',
             'anonymity', 'leave', 'mental_consequences', 'physical_consequences',
             'coworker_comfort', 'supervisor_comfort', 'equal_priority', 'observed_consequences']
    x = population.iloc[:, feature_indices].copy()
    x.columns = names
    # Preserve uncertainty and N/A as responses, not as negative answers.
    # Actual remaining blanks get a separate category; no ordinal assumptions.
    x = x.apply(lambda s: s.astype('string').str.strip().fillna('Missing response'))
    pd.DataFrame({'feature': names, 'source_index': feature_indices,
                  'question': cols[feature_indices]}).to_csv(OUT / 'feature_dictionary.csv', index=False)
    coded = np.column_stack([pd.Categorical(x[c]).codes for c in x])
    distance = squareform(pdist(coded, metric='hamming'))
    encoded = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit_transform(x)
    labels = {}
    scores = []
    for method in ['average_linkage', 'kmeans']:
        for k in range(2, 7):
            y = hierarchical(distance, k) if method == 'average_linkage' else KMeans(n_clusters=k, n_init=30, random_state=SEED).fit_predict(encoded)
            sizes = np.bincount(y)
            sil = silhouette_score(distance, y, metric='precomputed')
            labels[method, k] = y
            scores.append({'method': method, 'k': k, 'silhouette_hamming': sil,
                           'smallest_cluster': int(sizes.min()), 'largest_cluster': int(sizes.max())})
    evaluation = pd.DataFrame(scores)
    # Predeclared operational constraint: no group below 5% of the cohort.
    eligible = evaluation[evaluation.smallest_cluster >= 0.05 * len(x)]
    best = eligible.sort_values('silhouette_hamming', ascending=False).iloc[0]
    method, k = best['method'], int(best['k'])
    y = labels[method, k]
    # Deterministic presentation labels ordered by decreasing group size.
    order = pd.Series(y).value_counts().index
    relabel = {int(old): new + 1 for new, old in enumerate(order)}
    groups = np.array([relabel[int(a)] for a in y])
    evaluation.to_csv(OUT / 'model_comparison.csv', index=False)
    # Refit on ten random 80% subsamples and compare shared participants.
    rng = np.random.default_rng(SEED)
    stability = []
    for repeat in range(10):
        idx = np.sort(rng.choice(len(x), int(0.8 * len(x)), replace=False))
        fit = hierarchical(distance[np.ix_(idx, idx)], k) if method == 'average_linkage' else KMeans(n_clusters=k, n_init=30, random_state=SEED + repeat).fit_predict(encoded[idx])
        stability.append(adjusted_rand_score(y[idx], fit))
    # Sensitivity: remove care_options, which has additional missing responses.
    without = np.delete(coded, 1, axis=1)
    d2 = squareform(pdist(without, metric='hamming'))
    e2 = OneHotEncoder(sparse_output=False).fit_transform(x.drop(columns='care_options'))
    alternative = hierarchical(d2, k) if method == 'average_linkage' else KMeans(n_clusters=k, n_init=30, random_state=SEED).fit_predict(e2)
    sensitivity = adjusted_rand_score(y, alternative)
    pca = PCA(svd_solver='full').fit(encoded)
    coordinates = pca.transform(encoded)[:, :2]
    retained90 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .90) + 1)
    reduced = pca.transform(encoded)[:, :retained90]
    reduced_labels = KMeans(n_clusters=k, n_init=30, random_state=SEED).fit_predict(reduced)
    reduced_ari = adjusted_rand_score(y, reduced_labels)
    # t-SNE is a visual diagnostic only; clusters never fitted to its coordinates.
    tsne_checks = []
    for perplexity in [30, 50]:
        embedding = TSNE(n_components=2, perplexity=perplexity, init='pca',
                         learning_rate='auto', random_state=SEED, max_iter=1000).fit_transform(encoded)
        score = trustworthiness(encoded, embedding, n_neighbors=10)
        tsne_checks.append({'perplexity':perplexity,'trustworthiness_k10':float(score)})
        fig, ax = plt.subplots(figsize=(8,5))
        for g in sorted(set(groups)):
            member=groups==g
            ax.scatter(*embedding[member].T,s=14,alpha=.45,label=f'Cluster {g}')
        ax.set(title=f't-SNE diagnostic (perplexity {perplexity})',xlabel='t-SNE 1',ylabel='t-SNE 2')
        ax.legend();fig.tight_layout();fig.savefig(FIG/f'tsne_{perplexity}.png',dpi=180);plt.close(fig)
    pd.DataFrame({'component': np.arange(1, len(pca.explained_variance_ratio_)+1),
                  'variance_ratio': pca.explained_variance_ratio_}).to_csv(OUT / 'pca_variance.csv', index=False)
    records = []
    for c in x:
        for answer in sorted(x[c].unique()):
            overall = x[c].eq(answer).mean()
            for group in sorted(set(groups)):
                member = groups == group
                n = int((x.loc[member, c] == answer).sum())
                records.append({'cluster': group, 'feature': c, 'answer': answer,
                                'count': n, 'denominator': int(member.sum()),
                                'pct': 100*n/member.sum(), 'overall_pct': 100*overall,
                                'difference_pp': 100*(n/member.sum()-overall)})
    profile = pd.DataFrame(records)
    profile.to_csv(OUT / 'cluster_profiles.csv', index=False)
    context = []
    for group in sorted(set(groups)):
        p = population.loc[groups == group]
        context.append({'cluster': int(group), 'n': len(p), 'pct':100*len(p)/len(x),
                        'current_condition_yes_pct':100*p.iloc[:,47].eq('Yes').mean(),
                        'ever_treatment_yes_pct':100*p.iloc[:,52].eq(1).mean(),
                        'benefits_yes_pct':100*p.iloc[:,4].eq('Yes').mean(),
                        'supervisor_comfort_yes_pct':100*p.iloc[:,13].eq('Yes').mean(),
                        'resources_yes_pct':100*p.iloc[:,7].eq('Yes').mean()})
    pd.DataFrame(context).to_csv(OUT / 'cluster_summary.csv', index=False)
    # No respondent-level outputs are distributed; only aggregates.
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(8,5))
    for g in sorted(set(groups)):
        member = groups == g
        ax.scatter(*coordinates[member].T, s=15, alpha=.45, label=f'Cluster {g} (n={member.sum()})')
    ax.set(xlabel=f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)',
           ylabel=f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)',
           title='Workplace response profiles: two-dimensional PCA view')
    ax.legend(); fig.tight_layout(); fig.savefig(FIG/'pca_clusters.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4))
    for m, part in evaluation.groupby('method'):
        ax.plot(part.k, part.silhouette_hamming, marker='o', label=m.replace('_',' '))
    ax.set(xlabel='Number of clusters', ylabel='Silhouette (Hamming distance)', title='Comparison in the original categorical feature space')
    ax.legend(); fig.tight_layout(); fig.savefig(FIG/'model_comparison.png',dpi=180); plt.close(fig)
    yes = profile[profile.answer.eq('Yes')].pivot(index='feature',columns='cluster',values='pct')
    fig, ax = plt.subplots(figsize=(6,6))
    im=ax.imshow(yes.values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_yticks(range(len(yes)),yes.index)
    ax.set_xticks(range(len(yes.columns)),[f'Cluster {g}' for g in yes.columns])
    for i in range(len(yes)):
        for j in range(len(yes.columns)):
            ax.text(j,i,f'{yes.iloc[i,j]:.0f}%',ha='center',va='center',color='white' if yes.iloc[i,j]>60 else 'black')
    ax.set_title('Respondents answering Yes within each cluster')
    fig.colorbar(im,ax=ax,label='Percent'); fig.tight_layout(); fig.savefig(FIG/'cluster_profiles.png',dpi=180);plt.close(fig)
    summary = {'raw_rows':len(raw),'raw_columns':len(cols),'self_employed_excluded':int(raw.iloc[:,0].eq(1).sum()),
               'nontech_employees_excluded':int((raw.iloc[:,0].eq(0)&~mask).sum()),'cohort_n':len(x),
               'features':names,'onehot_columns':encoded.shape[1], 'cohort_missing_by_feature': x.eq('Missing response').sum().to_dict(),
               'selected_method':method,'selected_k':k,'silhouette':float(best.silhouette_hamming),
               'stability_ari':stability,'stability_mean':float(np.mean(stability)),
               'drop_care_options_ari':float(sensitivity),'pca_2d_variance':float(pca.explained_variance_ratio_[:2].sum()),
               'components_for_90pct':retained90,'sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
               'pca90_kmeans_ari':float(reduced_ari),'tsne_checks':tsne_checks,
               'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                           'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2)); print(pd.DataFrame(context).to_string(index=False))

if __name__ == '__main__':
    main()
