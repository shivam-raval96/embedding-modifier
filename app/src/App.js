import './App.css';
import {useState } from 'react';
import axios from "axios";
import SendIcon from '@mui/icons-material/Send';
import IconButton from '@mui/material/IconButton';
import CircularProgress from './progress'
import Scatterplot  from './plotdata'
import ScatterplotImg from './plotdataImg';
import data from './datasets/art_.json'
//import data_labels from './datasets/data2__labels.json'




const localDevURL = "http://127.0.0.1:8000/";
axios.defaults.headers.post['Content-Type'] ='application/json;charset=utf-8';
axios.defaults.headers.post['Access-Control-Allow-Origin'] = '*';

//let data_all ={"baseline":data,"time":frankenstein_time, "emotions":frankenstein_emotions, "characters":frankenstein_characters }
//let labels_all ={"baseline":data_labels,"time":frankenstein_time_labels, "emotions":frankenstein_emotions_labels, "characters":frankenstein_characters_labels}
function App() {

  const [plottedData, setPlottedData] = useState(data);
  const [dataset, setDataset] = useState('8attrLarge');
  const [labelData, setLabelData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [colorCol, setColorCol] = useState(4);
  const [jitter, setJitter] = useState(false)

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
    setLoading(true)
    let req = {
        dataset: dataset,
        theme: theme,
      };
    if (!preset.includes(theme)){
      axios
      .post(localDevURL + "modify-embeddings", req)
      .then((response) => {
        console.log(response.data.embeddings)
        setPlottedData(response.data.embeddings)
        setLabelData(response.data.labels )
        setLoading(false)
      })
      .catch((error) => {
        alert("Error in the backend");
        setLoading(false)
      });
    }else{

      loadData(dataset)
      
      
    }
    

  };

  /*
    const handleSend = () => {
    setPlottedData(data_all[theme])
    setLabelData(labels_all[theme])

  }
  
    
 */



  return (
    <div className="App">
      
      <div style={{ position: 'fixed', top: 20, left: 20, width: '350px', height:'380px', backgroundColor: 'rgba(0, 0, 0, 0.1)',
                 boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.5)',  // Drop shadow
                borderRadius: '20px' ,                         // Curved edges
                fontFamily: 'Perpetua'  // Setting the font family

            }}>

        
        <h2>Instruct to transform embeddings</h2>

       Dataset: <select 
        value={dataset} 
        onChange={e => {setDataset(e.target.value);theme='';setTheme('');console.log(e.target.value);loadData(e.target.value)}}
        style={{ width: '40%', padding: '5px', borderRadius: '5px' }}>
        <option value="data">Synth200</option>
        <option value="art">Art Data</option>
        <option value="hp">Harry Potter</option>

      </select>
      <br/>
      Color by: &nbsp;
         <select 
        value={colorCol} 
        onChange={e => {setColorCol(e.target.value)}}
        style={{ width: '37%', padding: '5px', borderRadius: '5px' }}>
        <option value="-1">DBSCAN Clusters</option>

        <option value="4">Colors</option>
        <option value="5">Animals</option>

        <option value="6">Places</option>

        <option value="7">Time</option>
        <option value="8">Emotion</option>
        <option value="9">Characters</option>
        <option value="10">Actions</option>
        <option value="11">Lit Style</option>

      </select>
      
      <br/>

  
       <label for="theme-choice"><b>Group by:</b> </label>
        <input list="theme-options" id="theme-choice" name="theme-choice" type="search"

        style={{
        boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.5)',  // Drop shadow
       borderRadius: '20px' ,                         // Curved edges
       fontFamily: 'Perpetua',  // Setting the font family
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
         
         <IconButton aria-label="send">

            {(loading)?<CircularProgress size="1.5rem"  color="inherit"style={{}}/>:<SendIcon onClick={handleSend}/>}
         
       </IconButton>
            <br/>
            <br/>
            <br/>
            <br/>
            <br/>

            <button  style={{ margin:"15px", padding: '5px'}}onClick={handleDownload}>Save View</button>


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
            <ScatterplotImg data={plottedData} labels ={labelData} colorCol ={colorCol} jitter = {jitter} width={1000} height={800} />

        </div>


        <div style={{ position: 'fixed', top: '40%', left: 20, width: '300px', height:'500px', backgroundColor: 'rgba(0, 0, 0, 0.02)',
                 boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.5)',  // Drop shadow
                borderRadius: '20px' ,                         // Curved edges
                fontFamily: 'Perpetua',  // Setting the font family
                overflowY: 'scroll'// <Scatterplot data={plottedData} labels ={labelData} colorCol ={colorCol} jitter = {jitter} width={1000} height={800} />

                

            }}><h3>Selection</h3>
            <div id = "selectioncontent" style={{ padding:'5px', }} ></div>
              <table>
                <tbody id="myTable">

                </tbody>
              </table>
            </div>


    </div>
  );
}

export default App;
