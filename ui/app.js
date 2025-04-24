const { useState, useEffect } = React;
const { 
  Search, Info, Film, Tv, AlertCircle, 
  TrendingUp, Clock, Award, BarChart, Zap 
} = lucide.icons;

const API_BASE_URL = 'http://localhost:8080/api';

const XGBoostRecommendationApp = () => {
  const [loading, setLoading] = useState(true);
  const [initialLoading, setInitialLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [sampleTitles, setSampleTitles] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [selectedTitle, setSelectedTitle] = useState(null);
  const [platforms, setPlatforms] = useState([]);
  const [contentTypes, setContentTypes] = useState([]);
  const [selectedPlatform, setSelectedPlatform] = useState('All');
  const [selectedType, setSelectedType] = useState('All');
  const [message, setMessage] = useState('');
  const [stats, setStats] = useState(null);
  const [activeView, setActiveView] = useState('sample'); 
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);
  const [modelInfo, setModelInfo] = useState({ name: 'XGBoost Hybrid' });
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setInitialLoading(true);
        setError('');
        
        const platformsResponse = await fetch(`${API_BASE_URL}/platforms`);
        if (!platformsResponse.ok) {
          throw new Error('Failed to fetch platforms');
        }
        const platformsData = await platformsResponse.json();
        if (platformsData.platforms) {
          setPlatforms(platformsData.platforms);
        }
        
        const typesResponse = await fetch(`${API_BASE_URL}/content_types`);
        if (!typesResponse.ok) {
          throw new Error('Failed to fetch content types');
        }
        const typesData = await typesResponse.json();
        if (typesData.content_types) {
          setContentTypes(typesData.content_types);
        }
        
        await fetchSampleTitles();
        
        const statsResponse = await fetch(`${API_BASE_URL}/stats`);
        if (!statsResponse.ok) {
          throw new Error('Failed to fetch statistics');
        }
        const statsData = await statsResponse.json();
        setStats(statsData);
        if (statsData.model_info) {
          setModelInfo(statsData.model_info);
        }
        
        setInitialLoading(false);
      } catch (error) {
        console.error('Error fetching initial data:', error);
        setError('Error connecting to the recommendation service. Please check if the API is running on port 8080.');
        setInitialLoading(false);
      }
    };
    
    fetchInitialData();
  }, []);

  const fetchSampleTitles = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(
        `${API_BASE_URL}/sample_titles?platform=${selectedPlatform}&type=${selectedType}`
      );
      
      if (!response.ok) {
        throw new Error('Failed to fetch sample titles');
      }
      
      const data = await response.json();
      
      if (data.titles) {
        setSampleTitles(data.titles);
        setActiveView('sample');
        setMessage('');
      } else {
        setMessage('No titles found with the selected filters.');
      }
    } catch (error) {
      console.error('Error fetching sample titles:', error);
      setError('Error loading titles. Please check if the API is running correctly.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!initialLoading) {
      fetchSampleTitles();
    }
  }, [selectedPlatform, selectedType]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    setActiveView('search');
    setError('');
    
    try {
      const response = await fetch(
        `${API_BASE_URL}/search?query=${encodeURIComponent(searchQuery)}&platform=${selectedPlatform}&type=${selectedType}`
      );
      
      if (!response.ok) {
        throw new Error('Search request failed');
      }
      
      const data = await response.json();
      
      if (data.results && data.results.length > 0) {
        setSearchResults(data.results);
        setMessage('');
      } else {
        setSearchResults([]);
        setMessage('No titles found matching your search criteria. Try a different query or adjust your filters.');
      }
    } catch (error) {
      console.error('Error searching titles:', error);
      setError('Error searching for titles. Please check if the API is running correctly.');
    } finally {
      setLoading(false);
    }
  };

  const getRecommendations = async (title) => {
    setSelectedTitle(title);
    setLoadingRecommendations(true);
    setError('');
    
    try {
      const response = await fetch(
        `${API_BASE_URL}/recommendations?id=${encodeURIComponent(title.id)}&title=${encodeURIComponent(title.title)}`
      );
      
      if (!response.ok) {
        throw new Error('Recommendation request failed');
      }
      
      const data = await response.json();
      
      if (data.recommendations) {
        setRecommendations(data.recommendations);
        setMessage('');
      } else {
        setRecommendations([]);
        setMessage('No recommendations found for this title.');
      }
    } catch (error) {
      console.error('Error getting recommendations:', error);
      setError('Error loading recommendations. Please check if the API is running correctly.');
    } finally {
      setLoadingRecommendations(false);
    }
  };

  const resetSearch = () => {
    setSearchQuery('');
    setSearchResults([]);
    setActiveView('sample');
    setError('');
  };

  const PlatformIcon = ({ platform }) => {
    const iconStyle = {
      width: 16,
      height: 16,
      borderRadius: '50%',
      display: 'inline-block',
      marginRight: 6,
      backgroundColor: 
        platform === 'Netflix' ? '#E50914' :
        platform === 'Amazon Prime' ? '#00A8E1' : 
        platform === 'Hulu' ? '#3DBB3D' : '#888'
    };
    
    return <span style={iconStyle}></span>;
  };

  const StatsDisplay = ({ stats }) => {
    if (!stats) return null;
    
    return (
      <div className="mt-4 p-3 bg-gray-50 rounded border">
        <h3 className="text-sm font-semibold mb-2">Dataset Statistics</h3>
        <div className="text-xs">
          <div><strong>Total Titles:</strong> {stats.total_titles?.toLocaleString()}</div>
          
          {stats.platforms && Object.keys(stats.platforms).length > 0 && (
            <div className="mt-1">
              <strong>Platforms:</strong>
              <ul className="list-disc pl-5 mt-1">
                {Object.entries(stats.platforms).map(([platform, count]) => (
                  <li key={platform}>{platform}: {count.toLocaleString()}</li>
                ))}
              </ul>
            </div>
          )}
          
          {stats.content_types && Object.keys(stats.content_types).length > 0 && (
            <div className="mt-1">
              <strong>Content Types:</strong>
              <ul className="list-disc pl-5 mt-1">
                {Object.entries(stats.content_types).map(([type, count]) => (
                  <li key={type}>{type}: {count.toLocaleString()}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-gray-800 text-white p-4">
        <div className="flex items-center justify-center">
          <h1 className="text-2xl font-bold text-center mr-2">XGBoost Streaming Recommendation System</h1>
          <Zap size={24} className="text-yellow-400" />
        </div>
      </header>
      
      {/* Main content */}
      <main className="flex flex-1 p-4 gap-4 overflow-auto">
        {/* Left sidebar - Filters */}
        <div className="w-64 bg-white p-4 rounded shadow flex flex-col">
          <h2 className="text-lg font-semibold mb-4">Filters</h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Platform</label>
            <select 
              className="w-full p-2 border rounded"
              value={selectedPlatform}
              onChange={(e) => setSelectedPlatform(e.target.value)}
              disabled={initialLoading}
            >
              <option value="All">All Platforms</option>
              {platforms.map(platform => (
                <option key={platform} value={platform}>{platform}</option>
              ))}
            </select>
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Content Type</label>
            <select 
              className="w-full p-2 border rounded"
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              disabled={initialLoading}
            >
              <option value="All">All Types</option>
              {contentTypes.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>
          
          <div className="flex gap-2 mb-4">
            <button 
              className="flex-1 bg-blue-600 text-white py-2 px-3 rounded text-sm flex items-center justify-center"
              onClick={fetchSampleTitles}
              disabled={initialLoading || loading}
            >
              <TrendingUp size={16} className="mr-1" />
              Apply Filters
            </button>
            {activeView === 'search' && (
              <button 
                className="flex-1 bg-gray-500 text-white py-2 px-3 rounded text-sm flex items-center justify-center"
                onClick={resetSearch}
                disabled={initialLoading}
              >
                <Clock size={16} className="mr-1" />
                Show Samples
              </button>
            )}
          </div>
          
          {/* Model info display */}
          <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
            <h3 className="text-sm font-semibold flex items-center mb-1">
              <BarChart size={16} className="mr-1 text-yellow-600" />
              Model Information
            </h3>
            <div className="text-xs">
              <p className="font-medium">{modelInfo.name || 'XGBoost Hybrid'}</p>
              <p className="text-gray-600 mt-1">{modelInfo.description || 'A hybrid recommendation system combining XGBoost and content-based filtering'}</p>
            </div>
          </div>
          
          {/* Platform stats */}
          <div className="mt-auto">
            <h3 className="text-md font-semibold mb-2">Platform Breakdown</h3>
            {stats && stats.platforms ? (
              <div className="space-y-2">
                {Object.entries(stats.platforms).map(([platform, count]) => (
                  <div key={platform} className="flex items-center">
                    <PlatformIcon platform={platform} />
                    <span>{platform}: {count.toLocaleString()} titles</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500 text-sm">Loading platform stats...</div>
            )}
          </div>
          
          <StatsDisplay stats={stats} />
        </div>
        
        {/* Center content - Search and results */}
        <div className="flex-1 flex flex-col gap-4">
          {/* Search bar */}
          <div className="bg-white p-4 rounded shadow">
            <div className="flex items-center">
              <input
                type="text"
                className="flex-1 p-2 border rounded-l"
                placeholder="Search for a movie or TV show..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                disabled={initialLoading}
              />
              <button 
                className="bg-blue-600 text-white p-2 rounded-r flex items-center"
                onClick={handleSearch}
                disabled={initialLoading || !searchQuery.trim()}
              >
                <Search size={20} />
              </button>
            </div>
          </div>
          
          {error && (
            <div className="bg-red-50 p-4 rounded shadow border border-red-200">
              <div className="flex items-start">
                <AlertCircle size={20} className="text-red-500 mr-2 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-medium text-red-800">Connection Error</h3>
                  <p className="text-sm text-red-700 mt-1">{error}</p>
                  <p className="text-xs text-red-600 mt-2">Make sure the API is running on port 8080. Check if recommendation_api.py is running correctly.</p>
                </div>
              </div>
            </div>
          )}
          
          {initialLoading ? (
            <div className="flex-1 bg-white p-4 rounded shadow flex items-center justify-center">
              <div className="text-center">
                <div className="w-16 h-16 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4 loading-spinner"></div>
                <p>Loading streaming content database...</p>
                <p className="text-sm text-gray-500 mt-2">Initializing XGBoost model and content features</p>
              </div>
            </div>
          ) : (
            <>
              {/* Results or sample titles */}
              <div className="bg-white p-4 rounded shadow flex-1 overflow-auto">
                <h2 className="text-lg font-semibold mb-4">
                  {activeView === 'search' ? 'Search Results' : 'Sample Titles'}
                </h2>
                
                {message && (
                  <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded flex items-start">
                    <AlertCircle size={20} className="text-yellow-500 mr-2 flex-shrink-0 mt-0.5" />
                    <p className="text-sm">{message}</p>
                  </div>
                )}
                
                {loading ? (
                  <div className="flex justify-center py-6">
                    <div className="w-10 h-10 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin loading-spinner"></div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {(activeView === 'search' ? searchResults : sampleTitles).map(item => (
                      <div 
                        key={item.id}
                        className={`p-3 border rounded cursor-pointer hover:bg-gray-50 transition-colors ${selectedTitle && selectedTitle.id === item.id ? 'border-blue-500 bg-blue-50' : ''}`}
                        onClick={() => getRecommendations(item)}
                      >
                        <div className="flex items-center mb-1">
                          <PlatformIcon platform={item.platform} />
                          <span className="text-sm text-gray-600">{item.platform}</span>
                        </div>
                        <h3 className="font-medium">{item.title}</h3>
                        <div className="flex justify-between text-sm text-gray-500 mt-1">
                          <span className="flex items-center">
                            {item.type === 'Movie' ? 
                              <Film size={14} className="mr-1" /> : 
                              <Tv size={14} className="mr-1" />
                            }
                            {item.type}
                          </span>
                          <span>{item.year}</span>
                        </div>
                        {item.genres && (
                          <div className="text-xs text-gray-500 mt-1 truncate">
                            {item.genres}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
        
        {/* Right sidebar - Recommendations */}
        <div className="w-72 bg-white p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-4 flex items-center">
            <Zap size={18} className="text-yellow-500 mr-2" />
            XGBoost Recommendations
          </h2>
          
          {loadingRecommendations ? (
            <div className="flex justify-center py-6">
              <div className="w-10 h-10 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin loading-spinner"></div>
            </div>
          ) : selectedTitle ? (
            <>
              <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded">
                <p className="text-sm text-gray-600">Based on your selection:</p>
                <p className="font-medium">{selectedTitle.title}</p>
                <div className="flex items-center mt-1">
                  <PlatformIcon platform={selectedTitle.platform} />
                  <span className="text-sm">{selectedTitle.platform} • {selectedTitle.type} • {selectedTitle.year}</span>
                </div>
              </div>
              
              {recommendations.length > 0 ? (
                <div className="space-y-3 overflow-auto max-h-[calc(100vh-320px)]">
                  {recommendations.map((rec, index) => (
                    <div key={index} className="p-3 border rounded">
                      <div className="flex items-center mb-1">
                        <PlatformIcon platform={rec.platform} />
                        <span className="text-sm text-gray-600">{rec.platform}</span>
                      </div>
                      <h3 className="font-medium">{rec.title}</h3>
                      <div className="flex justify-between text-sm mt-1">
                        <span className="text-gray-500">{rec.type} • {rec.year}</span>
                      </div>
                      <div className="mt-2 flex flex-col">
                        {rec.hybrid_score && (
                          <div className="flex items-center mb-1">
                            <div className="h-2 bg-gray-200 rounded-full w-full mr-2">
                              <div 
                                className="h-2 bg-green-500 rounded-full" 
                                style={{width: `${Math.min(100, Math.round(rec.hybrid_score * 100))}%`}}
                              />
                            </div>
                            <span className="text-xs font-medium whitespace-nowrap">{Math.round(rec.hybrid_score * 100)}% XGBoost</span>
                          </div>
                        )}
                        {rec.similarity && (
                          <div className="flex items-center">
                            <div className="h-2 bg-gray-200 rounded-full w-full mr-2">
                              <div 
                                className="h-2 bg-blue-500 rounded-full" 
                                style={{width: `${Math.min(100, Math.round(rec.similarity * 100))}%`}}
                              />
                            </div>
                            <span className="text-xs whitespace-nowrap">{Math.round(rec.similarity * 100)}% Content</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500">No recommendations found for this title.</p>
              )}
              
              <div className="mt-4 p-3 bg-gray-50 border rounded text-xs">
                <h3 className="font-semibold mb-1">About XGBoost Recommendations</h3>
                <p className="mb-2">
                  These recommendations use a hybrid approach combining:
                </p>
                <ul className="list-disc pl-4 space-y-1">
                  <li>XGBoost ML model feature importance</li>
                  <li>Content-based similarity (TF-IDF)</li>
                  <li>Genre and platform weighting</li>
                  <li>Release year proximity</li>
                </ul>
                <p className="mt-2 text-gray-600 italic">
                  The green score shows the hybrid recommendation strength while the blue score shows content similarity only.
                </p>
              </div>
            </>
          ) : (
            <div className="text-center py-10">
              <Info size={32} className="mx-auto text-gray-400 mb-3" />
              <p className="text-gray-500">Select a title to get XGBoost-powered recommendations</p>
            </div>
          )}
        </div>
      </main>
      
      {/* Footer */}
      <footer className="bg-gray-800 text-white p-3 text-center text-sm">
        <p>XGBoost Streaming Content Recommendation System • Using Models/XGBoost.pkl</p>
      </footer>
    </div>
  );
};
ReactDOM.render(<XGBoostRecommendationApp />, document.getElementById('root'));