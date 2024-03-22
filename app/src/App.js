import './App.css';
import {useState } from 'react';
import axios from "axios";
import CancelIcon from '@mui/icons-material/Cancel';
import IconButton from '@mui/material/IconButton';
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
//let data_all ={"baseline":data,"time":frankenstein_time, "emotions":frankenstein_emotions, "characters":frankenstein_characters }
//let labels_all ={"baseline":data_labels,"time":frankenstein_time_labels, "emotions":frankenstein_emotions_labels, "characters":frankenstein_characters_labels}
function App() {

  const [plottedData, setPlottedData] = useState(data);
  const [dataset, setDataset] = useState('relatedworks');
  const [labelData, setLabelData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [colorCol, setColorCol] = useState(4);
  const [jitter, setJitter] = useState(false)
  const [progress, setProgress] = useState(0)
  const [mapping, setMapping] = useState({})
  const [batchSize, setBatchSize] = useState(10)
  const [hoveredRowIndex, setHoveredRowIndex] = useState(null);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'ascending' });


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
        setMapping(data.mapping)
      }
      


      if (data.status && data.status === "Completed") {
          console.log(data.message);
          eventSource.close();
          setProgress(100)
          setLoading(false);

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
  const Legend = ({ stringToNumberMap, colors }) => {
    // Convert the object to an array of its values (names) for existing legend items
    const labels = Object.values(stringToNumberMap);
  
    return (
  <div style={{
        padding: '10px',
        border: '1px solid #ccc',
        borderRadius: '5px',
        backgroundColor: '#fff',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        fontFamily: 'Arial, sans-serif',
        fontSize: '20px',
        overflowY: 'scroll',
      }}>
        <h3 style={{ textAlign: 'left' }}>Legend</h3>
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
        <div style={{ display: 'flex', alignItems: 'left', marginBottom: '4px' }}>
          <span style={{
            display: 'inline-block',
            width: '20px',
            height: '20px',
            borderRadius: '50%',
            backgroundColor: '#F9F6EE', // Specific color for "not analysed"
            marginRight: '10px',
          }}></span>
          <span>Unprocessed</span>
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
  
  // Function to generate table rows from plottedData
  const generateTableRows = (data) => {
    return data.map((entry, index) => (
      <tr key={index} style={{cursor: 'context-menu'}}
        onMouseEnter={() => {
          setHoveredRowIndex(index);
          d3.selectAll('circle').transition(100).attr("r", function(d) {
            var dIsInSubset = d.id == entry.id;
            return dIsInSubset ? 15 : 5
          })
        }}
        onMouseLeave={() => setHoveredRowIndex(null)}>
        <td>{index + 1}</td>
        <td>{entry[2].slice(0, 200)}</td>
        <td>
          {/* Make this column editable */}
          <input 
            type="text" 
            value={entry[4]} 
            onChange={(e) => handleAttributeChange(index, e.target.value)}
            style={{width: '100%'}}
          />
        </td>
      </tr>
    ));
  };

  return (
    <div className="App">
      
      <div style={{ position: 'fixed', top: 20, left: 20, width: '300px', height:'450px', backgroundColor: 'rgba(0, 0, 0, 0.1)',
                 boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.5)',  // Drop shadow
                borderRadius: '20px' ,                         // Curved edges
                fontFamily: 'Arial, sans-serif',

            }}>

        
        <h2>Text Reprojector</h2>

       Dataset: <select 
        value={dataset} 
        onChange={e => {setDataset(e.target.value);theme='';setTheme('');console.log(e.target.value);loadData(e.target.value)}}
        style={{ width: '40%', padding: '5px', borderRadius: '5px' }}>
        <option value="8attrLarge">Synth200</option>
        <option value="poems">Poems</option>
        <option value="relatedworks">Related Works</option>
        <option value="art">Artworks</option>

        <option value="papers">Papers</option>
        <option value="greatgatsby">Great Gatsby</option>

      </select> 
      <br/> <br/> 
      Color by: &nbsp;
         <select 
        value={colorCol} 
        onChange={e => {setColorCol(e.target.value)}}
        style={{ width: '37%', padding: '5px', borderRadius: '5px' }}>
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
      
      <br/><br/> 

  
       <label for="theme-choice"><b>Reprojection attribute description:</b> </label><br/>
        <input list="theme-options" id="theme-choice" name="theme-choice" type="search"

        style={{
        boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.5)',  // Drop shadow
       borderRadius: '20px' ,                         // Curved edges
       fontFamily: 'Arial, sans-serif',
       overflowY: 'scroll',
       padding: '10px',
       fontSize: '16px', // Larger font size for better readability
       border: '2px solid #007bff', // Solid border with a color
       borderRadius: '10px', // Rounded corners
       color: '#495057', // Text color
       margin: '10px 0', // Margin to space out elements
       transition: 'border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out', // Smooth transition for focus
       
       }}

        onFocus={(e) => {
                e.target.style.borderColor = '#0056b3'; // Darker border on focus
                e.target.style.boxShadow = '0 0 0 0.2rem rgba(0, 123, 255, 0.5)'; // Glow effect on focus
              }}
        onBlur={(e) => {
                e.target.style.borderColor = '#007bff'; // Revert border color on blur
                e.target.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.5)'; // Revert box shadow on blur
              }}
                value={theme}
                    onChange={e => {setTheme(e.target.value)}}
                    />

        <datalist id="theme-options">
          <option value="colors"></option>
          <option value="animals"></option>
          <option value="places"></option>
          <option value="emotions"></option>
          <option value="time_of_day"></option>
          <option value="characters"></option>
          <option value="actions"></option>
          <option value="literary_styles"></option>
        </datalist>
         
        <button  style={{ margin:"15px", padding: '10px',fontSize: '15px', borderRadius: '10px'}}onClick={handleSend}>Transform</button>
        <input  id="batchsize" name="batchsize" value={batchSize} style={{ margin:"10px", padding: '10px',fontSize: '15px', width:"50px",borderRadius: '10px'}}onChange={e => {setBatchSize(e.target.value)}}type="number"/>
        <div style={{ padding: '10px',fontSize: '15px'}}>
        <Progress percent={progress} />

        </div>

         <IconButton aria-label="send">

            {(loading)?<CancelIcon size="1.5rem" variant="determinate" color="inherit" style={{}}onClick={handleCancel}/>:null}
         
       </IconButton>

            <button  style={{margin:"15px", padding: '10px',fontSize: '13px', borderRadius: '10px',cursor: 'pointer'}}onClick={handleDownload}>Save View</button>
            <input type="file" id="upload" style={{display: "none"}} accept=".json" onChange={handleUpload} />
            <label htmlFor="upload" style={{margin:"15px", padding: '10px',fontSize: '13px', borderRadius: '10px',border:'2px solid black',cursor: 'pointer'}}>Load View</label>



        <p style={{ paddingLeft: "15px",paddingRight: "15px", display:'none'}} align="left" >Each point is an embedded text. Visual clusters are identified by a clustering algorithm. <br /><br />
      This clustering may not be optimal for your task. You can change this!<br /><br />
        This view may not reflect the actual clustering in high dimensions. 
        
        </p>
        
        <div style={{display:'none'}}>
        <label id="name"><h4 style={{ paddingLeft: "15px",paddingTop: "0px"}} align="left">Load Projection</h4> </label>
        <input  type="file" accept=".json" onChange={handleFileChange} id="name" name="name" style={{ position:"relative", top: "-15px", left: "-10px"}} align="left" />
        <br />
        </div>
   
        </div>

        <div style={{ position: 'fixed', top: '0%', left: "15%",}}>
        {(dataset=='art')?
        <ScatterplotImg data={plottedData} labels ={labelData} colorCol ={colorCol} hoveredIndexTable={hoveredRowIndex} width={1400} height={700} />
        :<Scatterplot data={plottedData} labels ={labelData} colorCol ={colorCol} hoveredIndexTable={hoveredRowIndex} width={1400} height={700} />
        }
        </div>


        <div style={{ position: 'fixed', top: '50%', left: 20, width: '300px', height:'500px', backgroundColor: 'rgba(0, 0, 0, 0.02)',
                 boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.5)',  // Drop shadow
                borderRadius: '20px' ,                         // Curved edges
                fontFamily: 'Arial, sans-serif',display:'none',
                overflowY: 'scroll'// <Scatterplot data={plottedData} labels ={labelData} colorCol ={colorCol} jitter = {jitter} width={1000} height={800} />

                

            }}><h3>Selection</h3>
            <div id = "selectioncontent" style={{ padding:'5px', }} ></div>
              <table>
                <tbody id="myTable">

                </tbody>
              </table>
            </div>

            <div id = 'legend' style={{ position: 'fixed', top: '6%', left: '88%',
            maxwidth:'300px', 
                }}>
        
            <Legend stringToNumberMap={mapping} colors={colors} />

            
            </div>

            {/* New div for showing the data table */}
          <div style={{ position: 'fixed', top: '60%', left:"20%", width: '70%', height: '400px', backgroundColor: 'rgba(255, 255, 255, 0.9)', overflowY: 'auto', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.2)', borderRadius: '10px' }}>
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


    </div>
  );
}

export default App;
