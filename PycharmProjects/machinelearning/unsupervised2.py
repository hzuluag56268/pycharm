https://cml.ics.uci.edu/  ml repository
http://jse.amstat.org/jse_data_archive.htm

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.cluster_centers_
model.inertia_ #Number of inertia with the specified number of clusters x

from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler,kmeans)



from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize # not a transformer like Normilizer
#While StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) by removing the mean and scaling to unit variance, Normalizer() rescales each sample - here, each company's stock price - independently of the other.


'''Visualization with hierarchical clustering and t-SNE'''


from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
mergings = linkage(samples, method='single')
labels = fcluster(mergings,6, criterion='distance')
#


from sklearn.manifold import TSNE
model = TSNE(learning_rate=btw 50-200)



'''Decorrelating your data and dimension reduction'''
from sklearn.decomposition import NMF # for matrices or arrays,Non negative values
model = NMF(n_components=6) #n_components Must always be specified

from sklearn.feature_extraction.text import TfidfVectorizer # Gives sparse matrix
from sklearn.decomposition import TruncatedSVD # Used for sparse matrix
from sklearn.decomposition  import PCA
model = PCA()
#Apply the fit_transform method
# Get the mean of the grain samples: mean
mean = model.mean_
# Get the first principal component: first_pc
first_pc = model.components_[0,:]
# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

model.n_components_ #Provides the number of components
model.explained_variance_ #Provides the value of the variance to plot along with the N components

pca = PCA(n_components=2) #only keeps the two features with the highest variance

