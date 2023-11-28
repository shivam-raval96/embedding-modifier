import './App.css';
import {useState } from 'react';
import axios from "axios";
import SendIcon from '@mui/icons-material/Send';
import IconButton from '@mui/material/IconButton';
import CircularProgress from '@mui/material/CircularProgress';

import Scatterplot  from './plotdata'
import data from './datasets/data.json'
import data_labels from './datasets/data_labels.json'

import data_colors from './datasets/data_colors.json'
import data_colors_labels from './datasets/data_colors_labels.json'

import data_places from './datasets/data_places.json'
import data_places_labels from './datasets/data_places_labels.json'

import data_lit from './datasets/data_lit.json'
import data_lit_labels from './datasets/data_lit_labels.json'
/*import frankenstein from './datasets/frankenstein.json'
import frankenstein_labels from './datasets/frankenstein_labels.json'

import frankenstein_time from './datasets/frankenstein_time.json'
import frankenstein_time_labels from './datasets/frankenstein_time_labels.json'

import frankenstein_emotions from './datasets/frankenstein_emotions.json'
import frankenstein_emotions_labels from './datasets/frankenstein_emotions_labels.json'

import frankenstein_characters from './datasets/frankenstein_characters.json'
import frankenstein_characters_labels from './datasets/frankenstein_characters_labels.json'*/



const localDevURL = "http://127.0.0.1:8000/";
axios.defaults.headers.post['Content-Type'] ='application/json;charset=utf-8';
axios.defaults.headers.post['Access-Control-Allow-Origin'] = '*';

//let data_all ={"baseline":data,"time":frankenstein_time, "emotions":frankenstein_emotions, "characters":frankenstein_characters }
//let labels_all ={"baseline":data_labels,"time":frankenstein_time_labels, "emotions":frankenstein_emotions_labels, "characters":frankenstein_characters_labels}
let data_all ={"baseline":data, "colors":data_colors, "places":data_places, "lit":data_lit}
let labels_all ={"baseline":data_labels, "colors":data_colors_labels,  "places":data_places_labels, "lit":data_lit_labels}
function App() {

    
  
  const [plottedData, setPlottedData] = useState(data);
  const [labelData, setLabelData] = useState(data_labels);
  const [loading, setLoading] = useState(false);

  const [theme, setTheme] = useState('colors');


  const handleSend = () => {
    setPlottedData(data_all[theme])
    setLabelData(labels_all[theme])

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

  const handleDownload = () => {
      if (plottedData) {
          const blob = new Blob([JSON.stringify(plottedData, null, 2)], { type: 'application/json' });
          const link = document.createElement('a');
          link.href = URL.createObjectURL(blob);
          link.download = 'projections_'+theme+'.json';
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
      } else {
          alert('No JSON data to download');
      }
  };


  /*const handleSend = () => {
    setLoading(true)
    let req = {
        data: selectedOption,
        theme: theme,
      };
    
    axios
    .post(localDevURL + "modify-embeddings", req)
    .then((response) => {
      console.log(response.data.data)
      setPlottedData(frankenstein_emotions)
      setLabelData(frankenstein_emotions_labels)
      setLoading(false)
    })
    .catch((error) => {
      console.log(error);
      setLoading(false)
    });
    
  };*/



  return (
    <div className="App">
      
      <div style={{ position: 'fixed', top: 20, left: 20, width: '300px', backgroundColor: 'rgba(0, 0, 0, 0.1)',
                 boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.5)',  // Drop shadow
                borderRadius: '20px' ,                         // Curved edges
                fontFamily: 'Perpetua'  // Setting the font family

            }}>

        
        <h2>Embedding Projection Modifier</h2>

       Dataset: <select 
        value={'Frankenstein'} 
        onChange={null}
        style={{ width: '40%', padding: '5px', borderRadius: '5px' }}>
        <option value="frankstein">Frankenstein</option>
      </select>

      <button  style={{ margin:"15px", padding: '5px'}}onClick={handleDownload}>Save View</button>
      Cluster by: &nbsp;<input 
            style={{display:'none'}}
            type="text"
            value={theme}
            onChange={e => {setTheme(e.target.value)}}
         /> 
         <select 
        value={theme} 
        onChange={e => {setTheme(e.target.value)}}
        style={{ width: '40%', padding: '5px', borderRadius: '5px' }}>
        <option value="colors">Colors</option>
        <option value="places">Places</option>
        <option value="lit">Lit Style</option>

      </select>
         <IconButton aria-label="send">

            {(loading)?<CircularProgress size="1.5rem"  color="inherit"style={{}}/>:<SendIcon onClick={handleSend}/>}
         
       </IconButton>


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

        <div>
            <Scatterplot data={plottedData} labels ={labelData} width={1000} height={800} />
        </div>
    </div>
  );
}

export default App;
