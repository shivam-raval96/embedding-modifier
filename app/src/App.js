import './App.css';
import {useState, useEffect } from 'react';
import axios from "axios";
import CancelIcon from '@mui/icons-material/Cancel';
import IconButton from '@mui/material/IconButton';
import MapsUgcIcon from '@mui/icons-material/MapsUgc';
import SearchIcon from '@mui/icons-material/Search';
import Scatterplot  from './plotdata'
import ScatterplotImg from './plotdataImg';
import data from './datasets/relatedworks_.json'
//import data_labels from './datasets/data2__labels.json'
import { Progress } from 'react-sweet-progress';
import "react-sweet-progress/lib/style.css";
import * as d3 from 'd3';

const r_small = 5
const r_big = 15



const localDevURL = "http://127.0.0.1:8000/";
axios.defaults.headers.post['Content-Type'] ='application/json;charset=utf-8';
axios.defaults.headers.post['Access-Control-Allow-Origin'] = '*';


const colors = [
  "#d62728", "#bcbd22", "#000000", "#17becf", "#FFD700",
  "#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#25cfad",
  "#e377c2", "#8c564b", "#C28C9D", "#3498db", "#96FF2C",
  "#9b59b6", "#34495e", "#f1c40f", "#4F0000", "#C3BE7C",
  "#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#f39c12",
  "#16a085", "#2c3e50", "#7d3c98", "#c0392b", "#f7dc6f",
  "#48c9b0", "#f1948a", "#bb8fce", "#73c6b6", "#f0b27a",
  "#85c1e9", "#f7f9f9", "#720000", "#76448a"
];
const facets = [
  {
    "facet": "Artistic Movement or Style",
    "examples": [
      {
        "text": "Accent in Pink by Wassily Kandinsky",
        "attribute": "Abstract Art"
      },
      {
        "text": "Antibes by Claude Monet",
        "attribute": "Impressionism"
      }
    ]
  },
  {
    "facet": "Use of Color and Light",
    "examples": [
      {
        "text": "A Lady and Two Gentlemen by Rembrandt",
        "attribute": "Chiaroscuro"
      },
      {
        "text": "Alnwick Castle by William Turner",
        "attribute": "Expressive Colourisations"
      }
    ]
  },
  {
    "facet": "Thematic Content",
    "examples": [
      {
        "text": "Adoration of the Magi by Peter Paul Rubens",
        "attribute": "Religious Narratives"
      },
      {
        "text": "Archeological Reminiscence of Millet's Angelus by Salvador Dali",
        "attribute": "Surrealist Interpretation of Classical Works"
      }
    ]
  },
  {
    "facet": "Emotional or Conceptual Theme",
    "examples": [
      {
        "text": "Apparition of Face and Fruit Dish on a Beach by Salvador Dalí",
        "attribute": "Surrealism and Dreamlike Imagery"
      },
      {
        "text": "Ascent of the Blessed by Hieronymus Bosch",
        "attribute": "Moral and Religious Concepts"
      }
    ]
  },
  {
    "facet": "Historical or Mythological References",
    "examples": [
      {
        "text": "Ancient Rome; Agrippina Landing with the Ashes of Germanicus by William Turner",
        "attribute": "Historical Event"
      },
      {
        "text": "Atropos (The Fates) by Francisco Goya",
        "attribute": "Mythological Figures"
      }
    ]
  }
]
//let data_all ={"baseline":data,"time":frankenstein_time, "emotions":frankenstein_emotions, "characters":frankenstein_characters }
//let labels_all ={"baseline":data_labels,"time":frankenstein_time_labels, "emotions":frankenstein_emotions_labels, "characters":frankenstein_characters_labels}
function App() {

  const [plottedData, setPlottedData] = useState(data);
  const [dataset, setDataset] = useState('relatedworks');
  const [labelData, setLabelData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [colorCol, setColorCol] = useState(3);
  const [previousViews, setPreviousViews] = useState([]);
  const [previousAttributes, setPreviousAttributes] = useState({});
  const [attributes, setAttributes] = useState({})
  const [progress, setProgress] = useState(0)
  const [mapping, setMapping] = useState({})
  const [batchSize, setBatchSize] = useState(10)
  const [hoveredRowIndex, setHoveredRowIndex] = useState(null);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'ascending' });
  const [suggestedFacets, setSuggestedFacets] = useState(facets);
  const [searchQuery, setSearchQuery] = useState('Enter Query ...');

  useEffect(() => {
    const cache = localStorage.getItem('viewCache');
    if (cache) {
      const parsedCache = JSON.parse(cache);
      setPreviousViews(parsedCache.previousViews);
      setPreviousAttributes(parsedCache.previousAttributes);
    }
  }, []);


  var [theme, setTheme] = useState('');
  const [preset, setPreset] = useState(['','colors','animals','places','emotions', 'time_of_day', 'characters', 'actions','literary_styles']);

 const colorColCode={ colors:4, animals:5, places: 6, time_of_day:7, emotions: 8, characters: 9, actions: 10, literary_styles:11}

  const loadData = (dataset) => {

    try {
      //var data_labels2= require('./datasets/'+dataset+'_'+theme+'_labels.json')
      var data2= require('./datasets/'+dataset+'_'+theme+'.json');
  
  
      setPlottedData(data2)
      //setLabelData(data_labels2)
      if (theme!=''){
        setColorCol(colorColCode[theme])}
      setLoading(false)
  
     }
     catch (e) {
      console.log(e)
      setLoading(false)
      /*alert('Oops: Not Allowed')
      setDataset('small')
      setDR('umap')
      setClusterBy('content')*/
     }
  }

  const handleFileChange = (event) => {
      const file = event.target.files[0];
      if (file) {
          const reader = new FileReader();
          reader.onload = (e) => {
              try {
                  const json = JSON.parse(e.target.result);
                  setPlottedData(json)
              } catch (error) {
                  alert('Invalid JSON file');
              }
          };
          reader.readAsText(file);
      }
  };

  const handleUpload = (event) => {
    const fileReader = new FileReader();
    fileReader.readAsText(event.target.files[0], "UTF-8");
  
    fileReader.onload = e => {
      try {
        const parsedData = JSON.parse(e.target.result);
  
        setPlottedData(parsedData)
  
      } catch (error) {
        console.error("Error parsing JSON:", error);
        alert("Error parsing JSON. Please check the file format.");
      }
    };
  
    fileReader.onerror = e => {
      console.error("File reading error:", e);
      alert("Failed to read file. Please try again.");
    };
  };
  

  const handleDownload = () => {
      if (plottedData) {
          var blob = new Blob([JSON.stringify(plottedData, null, 2)], { type: 'application/json' });
          var link = document.createElement('a');
          link.href = URL.createObjectURL(blob);
          link.download = dataset+'_'+theme+'.json';
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);

          blob = new Blob([JSON.stringify(labelData, null, 2)], { type: 'application/json' });
          link.href = URL.createObjectURL(blob);
          link.download = dataset+'_'+theme+'_labels.json';
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);


      } else {
          alert('No JSON data to download');
      }
  };


  const handleSend = () => {
    setLoading(true);
    setProgress(0)
    const req = {
        dataset: dataset,
        theme: theme,
        batchsize: batchSize,
    };

    if (!preset.includes(theme)){

    axios.post('http://127.0.0.1:8000/initialize-embeddings', req)
    .then((response) => {
        console.log(response.data);
        const sessionId = response.data.session_id;
        listenForUpdates(sessionId); // Pass the session ID to the SSE connection function
    })
    .catch((error) => {
        console.error("Error initializing processing:", error);
        setLoading(false);
    });
  }else{

    loadData(dataset)
    
    
  }
  
};
const updateView = (newPlottedData, newAttributes) => {
  // Add current data to previousViews before updating
  if (plottedData.length > 0) {
    setPreviousViews(prevViews => [...prevViews, plottedData]);
    setPreviousAttributes(prevAttrs => ({ ...prevAttrs, ...attributes }));
  }
  
  // Update current view and attributes
  setPlottedData(newPlottedData);
  setAttributes(newAttributes);
};

function listenForUpdates(sessionId) {
  // Adjust the URL to include the session ID as a query parameter
  const eventSource = new EventSource(`http://127.0.0.1:8000/modify-embeddings/?session_id=${sessionId}`);

  eventSource.onmessage = function(event) {
      //console.log('Received update:', event.data);
        //setLabelData(response.data.labels )

      const data = JSON.parse(event.data);
      setProgress(parseInt(data.update*100))
      if (data.embeddings!='none'){
        setPlottedData(data.embeddings)
        //setLabelData(data.labels)
        attributes[theme]=JSON.parse(data.attributes)
        //console.log(attributes)
        setAttributes(attributes)
        //updateView(data.embeddings, attributes) 
        //saveCache()
        setMapping(data.mapping)
      }
      


      if (data.status && data.status === "Completed") {
          console.log(data.message);
          eventSource.close();
          setProgress(100)
          setLoading(false);
          //updateView(plottedData, attributes) 
          //saveCache()

      }
  };

  eventSource.onerror = function(error) {
      console.log('Error receiving updates:', error);
      eventSource.close();
      setProgress(0)
      setLoading(false);

  };
}

function handleCancel() {
  setProgress(100)
  axios.post('http://127.0.0.1:8000/stop-processing')
  .then((response) => {
      console.log(response.data);
      // Handle any UI changes needed after stopping the process
  })
  .catch((error) => {
      console.error("Error stopping processing:", error);
     

  });
}

const countStrings = (arr) => {
  const counts = {}; // Initialize an empty object to hold the counts

  arr.forEach((str) => {
    if (counts[str]) {
      counts[str] += 1; // Increment the count if the string is already in the object
    } else {
      counts[str] = 1; // Initialize with 1 if it's the first occurrence of the string
    }
  });

  return counts;
};


  const Legend = ({ stringToNumberMap, colors }) => {
    // Convert the object to an array of its values (names) for existing legend items
    const labels = Object.values(stringToNumberMap);
    
    
    //const resultMap = countStrings(attributes[theme]);
    //console.log(resultMap); // Converting the Map to an object for easier viewing
  
    return (
  <div className="legend-container">
        <h3 style={{ textAlign: 'left' }}>Attributes Found</h3>
        {labels.map((name, index) => (
          <div key={index} style={{
            display: 'flex', // Use flexbox for alignment
            alignItems: 'center', // Center items vertically
            marginBottom: '4px',
          }}>
            <span style={{
              display: 'inline-block',
              width: '20px',
              height: '20px',
              borderRadius: '50%',
              backgroundColor: colors[index % colors.length],
              marginRight: '10px', // Add some space between the circle and the text
            }}></span>
            <span style={{ flex: '1' }}>{name}</span> {/* This ensures the text takes the remaining space */}
          </div>
        ))}
        {/* Manually add the "not detected" and "not analysed" entries */}
        <div style={{ display: 'flex', alignItems: 'left', marginBottom: '4px', display:'none'}}>
          <span style={{
            display: 'inline-block',
            width: '20px',
            height: '20px',
            borderRadius: '50%',
            backgroundColor: '#808080', // Specific color for "not detected"
            marginRight: '10px',
          }}></span>
          <span>None</span>
          <br/><br/>
        </div>

      </div>
    );
  };

  const getSortedData = () => {
    if (!sortConfig.key) return plottedData;
  
    const sortedData = [...plottedData].sort((a, b) => {
      if (a[sortConfig.key] < b[sortConfig.key]) {
        return sortConfig.direction === 'ascending' ? -1 : 1;
      }
      if (a[sortConfig.key] > b[sortConfig.key]) {
        return sortConfig.direction === 'ascending' ? 1 : -1;
      }
      return 0;
    });
  
    return sortedData;
  };

  const requestSort = (key) => {
    let direction = 'ascending';
    if (sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    } else {
      direction = 'ascending';
    }
    setSortConfig({ key, direction });
  };
  
  
  const handleAttributeChange = (index, newValue) => {
    // Create a new array with all items but replace the item at the given index with a new object
    const newData = plottedData.map((item, i) => {
      if (i === index) {
        // Replace the last element (attribute) of the current entry with the new value
        const updatedItem = [...item];
        updatedItem[3] = newValue;
        return updatedItem;
      }
      return item;
    });
  
    // Update the plottedData state with the new array
    setPlottedData(newData);
  };

  // Save to local storage (simplified example)
  const saveCache = () => {
    const cache = {
      previousViews: previousViews,
      previousAttributes: previousAttributes,
    };
    localStorage.setItem('viewCache', JSON.stringify(cache));
  };

  const restoreView = (index) => {
    // Calculate the actual index in the previousViews array
    const actualIndex = previousViews.length - 1 - index;
  
    // Update the current view with the selected previous view
    // Ensure that this operation does not push the current view into the history again
    const restoredData = previousViews[actualIndex];
    const restoredAttributes = previousAttributes[actualIndex] || {};
  
    // Set the current view to the restored view
    setPlottedData(restoredData);
    setAttributes(restoredAttributes);

  };


  const handleTableDataGenration = () => {
    const texts = plottedData.map(sublist => {
      return { id: sublist, text: sublist[2]};
    });
  }
  
  // Function to generate table rows from plottedData
  const generateTableRows = (data) => {
    return data.map((entry, index) => (
      <tr key={index} style={{cursor: 'context-menu'}}
        onMouseEnter={() => {
          setHoveredRowIndex(index);
          d3.selectAll('circle').each(function(d) {
            // Capture the original size
            var originalR = +d3.select(this).attr('r');

            // Check if the current element's id matches the condition
            var isTarget = d.id === entry.id;
            
            // If it's the target, double its size, then transition back to the original size
            if (isTarget) {
              d3.select(this).raise()
              .transition().duration(500)
                .attr("r", originalR * 3)
                .transition().duration(500)  // Chain another transition to revert
                .attr("r", originalR)
            }
          });
          
          d3.selectAll('image').each(function(d) {
            // Capture the original size
            var originalWidth = +d3.select(this).attr('width');
            var originalHeight = +d3.select(this).attr('height');
            
            // Check if the current element's id matches the condition
            var isTarget = d.id === entry.id;
            
            // If it's the target, double its size, then transition back to the original size
            if (isTarget) {
              d3.select(this).raise()
              .transition().duration(500)
                .attr("width", originalWidth * 3)
                .attr("height", originalHeight * 3)
                .transition().duration(500)  // Chain another transition to revert
                .attr("width", originalWidth)
                .attr("height", originalHeight);
            }
          });
        }}
        onMouseLeave={() => setHoveredRowIndex(null)}>
        <td>{index + 1}</td>
      {entry.slice(2).map((cellValue, colIndex) => (
        <td key={colIndex}>
          {typeof cellValue === 'string' && cellValue.length > 200 ? (
                cellValue.slice(0, 200) + '...'
              ) : (
                <input className='data-attribute'
                  type="text"
                  value={cellValue}
                  onChange={(e) => handleAttributeChange(index, colIndex + 2, e.target.value)} // Adjust colIndex for handling changes
                  style={{ width: '100%', margin: '2px' }}
                />)}
        </td>
        ))}
        
      </tr>
    ));
  };



  const handleSuggestFacets = () => {
    const texts = plottedData.map(sublist => sublist[2]).slice(0, 10); // Assuming texts start at index 2

    const req = {
        texts: texts,
        
    };
    axios.post('http://127.0.0.1:8000/suggest-facets', req)
    .then((response) => {
        console.log(response.data);
        setSuggestedFacets(response.data.facets['facets'])
    })
    .catch((error) => {
        console.error("Error initializing processing:", error);
    });
  
  
};

  
  
  

  return (
    <div className="App">
      <div className="container">
        <div id="controls">
          <h3>Text Facets</h3>

          <div className="dropdown">
            <span>Dataset:</span> 
            <select
              value={dataset}
              // Stringing along multiple lines in this is terrible practice, this should be a function
              onChange={e => { setDataset(e.target.value); theme = ''; setTheme(''); loadData(e.target.value) }}
            >
              <option value="8attrLarge">Synth200</option>
              <option value="poems">Poems</option>
              <option value="art">Artworks</option>
              <option value="relatedworks">Related Works</option>
              <option value="papers">Papers</option>
              <option value="greatgatsby">Great Gatsby</option>
            </select>
          </div>

          <div className="dropdown" style={{display:'none'}}>
            <span>Color by:</span>
            <select
              value={colorCol}
              onChange={e => { setColorCol(e.target.value) }}
            >
              <option value="-1">GPT Clusters</option>
              <option value="4">Colors</option>
              <option value="5">Animals</option>
              <option value="6">Places</option>
              <option value="7">Time</option>
              <option value="8">Emotion</option>
              <option value="9">Characters</option>
              <option value="10">Actions</option>
              <option value="11">Lit Style</option>
            </select>
          </div>
          <br/>

          <div className="button-spread">
            <button className="control-button" onClick={handleDownload}>Save View</button>
            <input type="file" id="upload" style={{ display: "none" }} accept=".json" onChange={handleUpload} />
            <label className="control-button" htmlFor="upload">Load View</label>
          </div>

          <div style={{ display: 'none' }}>
            <label id="name"><h4 style={{ paddingLeft: "15px", paddingTop: "0px" }} align="left">Load Projection</h4> </label>
            <input type="file" accept=".json" onChange={handleFileChange} id="name" name="name" style={{ position: "relative", top: "-15px", left: "-10px" }} align="left" />
            <br />
          </div>

        </div>
        
        <Legend stringToNumberMap={mapping} colors={colors} />
        <div className="previous-views-container">
          <h3>Previous Views</h3>
          {previousViews.slice(-5).reverse().map((view, index) => (
            <div key={index} className="previous-view" onClick={() => restoreView(index)}>
              {/* Provide a way to identify or preview the view. Adjust as needed. */}
              <span>View {previousViews.length - index}</span>
            </div>
          ))}
        </div>
           
      </div>  

      <div className="scatterplot-selection-attribute-container">
        <div className="scatterplot-attribute-container">
          <div id="scatterplot">
            {(dataset=='art')?
            <ScatterplotImg data={plottedData} labels ={labelData} colorCol ={colorCol} attributes = {attributes} hoveredIndexTable={hoveredRowIndex} searchQuery = {searchQuery} width={1200} height={600} />
            :<Scatterplot data={plottedData} labels ={labelData} colorCol ={colorCol} attributes = {attributes} hoveredIndexTable={hoveredRowIndex}  searchQuery = {searchQuery} width={1200} height={600} />
            }
          </div>
          <div id ='attribute' className="attribute-container">
            <div>
              <label htmlFor="theme-choice">Reprojection attribute description:</label>
              <textarea cols="120" rows="6" className="attribute-description" id="theme-choice" name="theme-choice" 
                value={theme}
                onChange={e => { console.log(theme);setTheme(e.target.value) }}
              />
            </div>

            <div className="button-spread" id="transform">
              <button className="control-button" onClick={handleSend}>Transform</button>
              <IconButton aria-label="send">
              {(loading) ? <CancelIcon fontSize="large" variant="determinate" color="inherit" onClick={handleCancel} /> : null}
            </IconButton>
            </div>

            <div>
              <Progress percent={progress} />
            </div>


            
            <div className="suggest-container">
              <label htmlFor="suggest" className="suggest-label">Give suggestions:</label>
              <IconButton aria-label="suggest" className="suggest-icon-button">
                <MapsUgcIcon fontSize="large" variant="determinate" color="inherit" onClick={handleSuggestFacets} />
              </IconButton>
            </div>
            <div className="suggestions-container">
              {suggestedFacets.map((item) => (
                <div key={item.facet} className="suggestion-item">
                  <div className="facet-name">
                    {item.facet}
                  </div>
                  <div className="examples-container">
                    {item.examples.map((example, index) => (
                      <textarea
                        key={index}
                        readOnly
                        value={`${example.text} - ${example.attribute}`}
                        className="example-textarea"
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>


        </div>
        <div style={{ position: 'fixed', top: '1%', left: '40%', backgroundColor: 'rgba(0, 0, 0, 0.02)',
                 boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.5)',  // Drop shadow
                borderRadius: '40px' ,                         // Curved edges
                fontFamily: 'Arial, sans-serif',
                overflowY: 'scroll',
                padding: '10px',
                fontSize: '22px', // Larger font size for better readability
                borderRadius: '10px', // Rounded corners
                color: '#495057', // Text color                
                }}>
           <label for="search"><b>Search:</b> </label>

        <input id="search" name="search" style={{ fontSize: '20px',  }}
      type="text"
      placeholder="Search..."
      value={searchQuery}
      onChange={(e) => {console.log(e.target.value);return setSearchQuery(e.target.value)}}

    /></div>

        <div className="container"id="tableContainer" style={{ width: '100%' }}>
            <table>
              <tbody id="myTable">
              </tbody>
            </table>

          </div>

      </div>

          
    </div>

  );
}

export default App;

/*

              <div id="selection">
              <h3>Selection</h3>
              
              <div id="content"></div>
              <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th>#</th>
                    <th style={{cursor: 'ns-resize'}}onClick={() => requestSort(2)}>Text</th>
                    <th style={{cursor: 'ns-resize'}}onClick={() => requestSort(4)}>{theme}</th>
                  </tr>
                </thead>
                <tbody>
                  {generateTableRows(getSortedData())}
                </tbody>
              </table>
            </div>

>*/