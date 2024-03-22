import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { PieChart } from 'react-minimal-pie-chart';


const r_small = 5
const r_big = 15

const colors2 = [
    "#9edae5", "#17becf", "#dbdb8d", "#bcbd22", "#c7c7c7", "#7f7f7f", 
    "#f7b6d2", "#e377c2", "#c49c94", "#8c564b", "#c5b0d5", "#9467bd", 
    "#ff9896", "#d62728", "#98df8a", "#2ca02c", "#ffbb78", "#ff7f0e", 
    "#aec7e8", "#1f77b4"
]

const colors = ["#F9F6EE","#808080",
"#d62728", "#bcbd22", "#000000", "#17becf", "#FFD700",
"#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#25cfad",
"#e377c2", "#8c564b", "#C28C9D", "#3498db", "#96FF2C",
"#9b59b6", "#34495e", "#f1c40f", "#4F0000", "#C3BE7C",
"#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#f39c12",
"#16a085", "#2c3e50", "#7d3c98", "#c0392b", "#f7dc6f",
"#48c9b0", "#f1948a", "#bb8fce", "#73c6b6", "#f0b27a",
"#85c1e9", "#f7f9f9", "#720000", "#76448a"
]
function addAlpha(color, opacity) {
    // coerce values so ti is between 0 and 1.
    var _opacity = Math.round(Math.min(Math.max(opacity || 1, 0), 1) * 255);
    return color.slice(0,7) + _opacity.toString(16).toUpperCase();
}
const getColumn = (arr, n) => arr.map(x => x[n]);

function encodeArray(array) {
    // Step 1: Identify unique strings
    const uniqueStrings = new Set(array);


    // Step 2: Map strings to numbers
    const stringToNumberMap = {};
    let number = 0;
    uniqueStrings.forEach(str => {
        stringToNumberMap[str] = number++;
    });

    // Step 3: Generate encoded array
    return stringToNumberMap;
}

function getEncodedArray(array, stringToNumberMap){

    return array.map(str => stringToNumberMap[str]);

}

function getPieDistribution(data, labels) {
    let distribution = {};

    for (let i = 0; i < labels.length; i++) {
        let label = labels[i];
        let item = data[i];

        if (!distribution[label]) {
            distribution[label] = {};
        }

        if (!distribution[label][item]) {
            distribution[label][item] = 0;
        }

        distribution[label][item]++;
    }

    return distribution;
}


function trackPointer(e, { start, move, out, end }) {
    const tracker = {},
      id = (tracker.id = e.pointerId),
      target = e.target;
    tracker.point = d3.pointer(e, target);
    target.setPointerCapture(id);
  
    d3.select(target)
      .on(`pointerup.${id} pointercancel.${id} lostpointercapture.${id}`, (e) => {
        if (e.pointerId !== id) return;
        tracker.sourceEvent = e;
        d3.select(target).on(`.${id}`, null);
        target.releasePointerCapture(id);
        end && end(tracker);
      })
      .on(`pointermove.${id}`, (e) => {
        if (e.pointerId !== id) return;
        tracker.sourceEvent = e;
        tracker.prev = tracker.point;
        tracker.point = d3.pointer(e, target);
        move && move(tracker);
      })
      .on(`pointerout.${id}`, (e) => {
        if (e.pointerId !== id) return;
        tracker.sourceEvent = e;
        tracker.point = null;
        out && out(tracker);
      });
  
    start && start(tracker);
  }





function Scatterplot(props) {
  var { width, height, data , labels, colorCol, mapping} = props;
  const margin = {top:100, left:120, right:80, bottom:100}
  const ref = useRef();
  const [pieCharts, setPieCharts] = useState([])
  const defaultLasso = [[0,0]]
  const [prevData, setPrevData] = useState(null)
  const [tooltipIndices, setTooltipIndices] = useState([]);


  const [searchQuery, setSearchQuery] = useState('Enter Query ...');
  const [stringToNumberMap, setStringToNumberMap] = useState({'0':'none'});

  function selectIndices(data, gridRows, gridCols) {
    // Calculate grid cell size based on data bounds and desired grid dimensions
    const xMin = d3.min(data, d => d[0]);
    const xMax = d3.max(data, d => d[0]);
    const yMin = d3.min(data, d => d[1]);
    const yMax = d3.max(data, d => d[1]);
    const cellWidth = (xMax - xMin) / gridCols;
    const cellHeight = (yMax - yMin) / gridRows;
  
    // Initialize an empty grid
    let grid = Array.from({ length: gridRows }, () => Array.from({ length: gridCols }, () => []));
  
    // Assign points to the appropriate grid cells
    data.forEach((point, index) => {
      const col = Math.min(Math.floor((point[0] - xMin) / cellWidth), gridCols - 1);
      const row = Math.min(Math.floor((point[1] - yMin) / cellHeight), gridRows - 1);
      grid[row][col].push(index); // Store the index of the point
    });
  
    // Select one point from each occupied cell
    const selectedIndices = [];
    grid.forEach(row => row.forEach(cell => {
      if (cell.length > 0) {
        const randomIndex = cell[Math.floor(Math.random() * cell.length)];
        selectedIndices.push(randomIndex);
      }
    }));
  
    return selectedIndices;
  }
  


  useEffect(() => {
    if (data && data !== prevData) {
      setPrevData(data); // Save the current data as previous before it changes
    }


    data.forEach((d,i)=>{d.id = i})
    const svg = d3.select(ref.current);


    function lasso() {
        const dispatch = d3.dispatch("start", "lasso", "end");
        const lasso = function(selection) {
          const node = selection.node();
          const polygon = [];
      
          selection
            .on("touchmove", e => e.preventDefault()) // prevent scrolling
            .on("pointerdown", e => {
              trackPointer(e, {
                start: p => {
                  polygon.length = 0;
                  dispatch.call("start", node, polygon);
                },
                move: p => {
                  polygon.push(p.point);
                  dispatch.call("lasso", node, polygon);
                },
                end: p => {
                  dispatch.call("end", node, polygon);
                }
              });
            });
        };
        lasso.on = function(type, _) {
          return _ ? (dispatch.on(...arguments), lasso) : dispatch.on(...arguments);
        };
      
        return lasso;
      }

    //svg.selectAll('*').remove()
    svg.selectAll('text.label').remove()
    svg.selectAll("path").remove()
    d3.select("#selectioncontent").selectAll('*').remove()
    d3.selectAll(".line").remove();

    
    const path = d3.geoPath()
    const l = svg.append("path").attr("class", "lasso")

    svg.append("defs").append("style").text(`
    .selected {r: 2.5; fill: red}
    .lasso { fill-rule: evenodd; fill-opacity: 0.1; stroke-width: 1.5; stroke: #000; z-index: 0;}
  `);


    function draw(polygon) {
        //d3.select("#selectioncontent").selectAll('*').remove()

        // selectioncontent
       // const selectioncontent = d3.select("#selectioncontent").append("div")
        //var content = '<table>'

        let table = document.getElementById("myTable");
        table.innerHTML = "";

        l.datum({
          type: "LineString",
          coordinates: polygon
        }).attr("d", path).style('z-index',-10);
    
        const selected = [];
    
        // note: d3.polygonContains uses the even-odd rule
        // which is reflected in the CSS for the lasso shape
        circles.classed(
          "selected",
          polygon.length > 2
            ? d => {d3.polygonContains(polygon, [xScale(d[0]),yScale(d[1])]) && selected.push(d) }//&& (content += " <tr><td>"+d[2] + "</td></tr>" )}
            : false
        );
        //content += '</table>'

        /*for (const [key, idx] of Object.entries(stringToNumberMap)){
            console.log(key,"<font color=\"red\">"+key+"</font>")
            content = content.replaceAll(key,"<font color=\""+colors[idx]+"\">"+key+"</font>")
        }*/

        var TableBackgroundNormalColor = "#ffffff";
        var TableBackgroundMouseoverColor = "#b8b6b6";

        // These two functions need no customization.
        function ChangeBackgroundColor(row,id) { 
            row.style.backgroundColor =  row.style.backgroundColor.replace(/[^,]+(?=\))/, 0.8);

            row.style.border= '1px solid black !important'

            svg.selectAll('circle').transition(100).attr("r", function(d) {
                var dIsInSubset = d.id == id;
                return dIsInSubset ? r_big : r_small
              })
        
        }
        function RestoreBackgroundColor(row) { 
            row.style.backgroundColor =  row.style.backgroundColor.replace(/[^,]+(?=\))/, 0.4);
            //row.style.backgroundColor = TableBackgroundNormalColor; 
            svg.selectAll('circle').transition(100).attr("r", r_small)
        }

        selected.forEach(d=>{
            let tr = document.createElement("tr");
            tr.innerHTML =`<td>${d[2]}</td>`;
            tr.style.backgroundColor = addAlpha(colors[color_idx[d.id]],0.4)

            //tr.addEventListener('mouseover', () => console.log(d));
            tr.addEventListener('mouseover', () => {console.log(d);ChangeBackgroundColor(tr,d.id)});
            tr.addEventListener('mouseout', () => {console.log(d);RestoreBackgroundColor(tr)});

            table.appendChild(tr);

        })
        svg.value = { polygon, selected };

        //selectioncontent.html(content)
        //selectioncontent.addEventListener('mouseover', () => console.log(datum));



      }
    

    

    

    // Set scales for the scatterplot
    const xScale = d3.scaleLinear()
      .domain([d3.min(data, d => d[0]), d3.max(data, d => d[0])])
      .range([margin.left, width-margin.right]);

    const yScale = d3.scaleLinear()
      .domain([d3.min(data, d => d[1]), d3.max(data, d => d[1])])
      .range([height-margin.bottom, margin.top]);

      data = data.map(d => ({
        ...d, // Spread the rest of the data object to retain other properties
        scaledX: xScale(d[0]), // Apply xScale to the original x value
        scaledY: yScale(d[1])  // Apply yScale to the original y value
      }));


      if (tooltipIndices.length == 0) {
        var indices = selectIndices(data, 3, 2)
        console.log('hello',indices)
        setTooltipIndices(indices)}

    // Select random points using the indices
    const randomDataPoints = tooltipIndices.map(index => data[index]);
    console.log(randomDataPoints)

    // Function to create tooltip
    const createTooltip = (d) => {
      //console.log(d)

        const tooltipDiv = d3.select("body").append("div")
            .attr("class", `autotooltip tooltip-${d.id}`)
            .style("opacity", 0)
            .style("width", 200)
            .style("background-color", "white")
            .style("border", "solid")
            .style("border-width", "2px")
            .style("border-radius", "5px")
            .style("padding", "5px")
            .style("margin-right", "50px")
            .style("position", "absolute");

        // Set tooltip text and position
        tooltipDiv.html(d[2].slice(0, 50)+'...')
            .style("left", (xScale(d[0])+50) + "px")
            .style("top", (yScale(d[1])-35) + "px")
            .transition()
            .duration(1500)
            .style("opacity", 1);

            
    };

    d3.selectAll(".autotooltip").remove();
    // Create and show tooltips for random points
    randomDataPoints.forEach(d => createTooltip(d));






    // Tooltips
    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0)
        .style("background-color", "white")
        .style("border", "solid")
        .style("border-width", "2px")
        .style("border-radius", "5px")
        .style("padding", "5px")
        .style("margin-right", "50px")
        .style("position", "absolute");

  //console.log(data)
    // Bind data to circles and add tooltips
    //console.log(labels)
    // Compute the density contours.
    if (false){

    
    const contours = d3.contourDensity()
    .x(d => xScale(d[0]))
    .y(d => yScale(d[1]))
    .size([width, height])
    .bandwidth(20)
    .thresholds(5)
    (data);



    //console.log(contours)
    

    // Append the contours.
    var contourPaths = svg.selectAll('contours').attr("stroke-linejoin", "round")
        .attr("stroke", 'black')
        .data(contours)
        .join("path")
        .style("opacity", 0) 
        .attr("fill", (d, i) => 'gray')
        .attr("stroke-width", (d, i) => i % 5 ? 0.25 : 1)
        //.attr("stroke", 'red')
        .style("z-index", -2)
        .attr("d", d3.geoPath());

    contourPaths.transition().duration(2500).style("opacity", (d, i) => i % 5 ? 0 : 0.2)}

    //SetcolorCol
    if (colorCol!=-1){
        var array = getColumn(data,colorCol)
        var stringToNumberMap = encodeArray(array) ;
        var color_idx = getEncodedArray(array, stringToNumberMap).map(number => number + 2);




    }else{
        var color_idx = getColumn(data,3)

    }



     // Bind data to circles
     const circles = svg.selectAll('circle').data(data);

     // Enter new circles
     circles.enter().append('circle')
         .attr('cx', d => xScale(d[0]))
         .attr('cy', d => yScale(d[1]))
         .attr('r', (d,i) => (d[2].toLowerCase().includes(searchQuery.toLowerCase())) ? r_big : r_small)
         .style("opacity", 0.9)
        .attr('fill', (d,i) =>  colors[color_idx[i]%40])
        .attr('stroke', 'black')  // Add this line for the boundary color
        .attr('stroke-width', 0.5)  // Add this line for the boundary width
        .style("z-index", 2)
        .on("mouseover", (event, d) => {
          d3.selectAll(".autotooltip").transition().duration(100).style("opacity", 0);

            svg.selectAll('circle')
            .transition().duration(100)
            .style("opacity", 0.9) 
            .attr('stroke-width', 0.5) 
            //.attr('r', r_small);
            d3.select(event.currentTarget).transition().duration(100)
                .style("opacity", 1) 
                .attr('stroke-width', 4)  // Add this line for the boundary width
                //.attr('r', r_small);

            let highlightedText = d[2];
            if (searchQuery) {
                  const regex = new RegExp(`(${searchQuery})`, 'gi');
                  highlightedText = highlightedText.replace(regex, "<span style='background-color: yellow;'>$1</span>");
                }

            tooltip.transition()
                .duration(100)
                .style("opacity", .9);
            tooltip.html(highlightedText.slice(0, 700))
                .style("left", (event.pageX + 5) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", (event,d) => {
          svg.selectAll('circle')
            .transition().duration(100)
            .style("opacity", 0.9) 
            .attr('stroke-width', 0.5) 

          // Show tooltips for random points again
        //d3.selectAll(".tooltip").remove(); // First remove all existing tooltips
        d3.selectAll(".autotooltip").transition().duration(100).style("opacity", 1);




           d3.select(event.currentTarget).transition().duration(100)
                .style("opacity", 0.9) 
                .attr('stroke-width', 0.5)  // Add this line for the boundary width*/

            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        })

      circles.attr('r', d => d[2].toLowerCase().includes(searchQuery.toLowerCase()) ? r_big : r_small); // Adjust the dot size based on the search query

     
     // Update existing circles
     circles.attr('fill', (d,i) =>  colors[color_idx[i]%40]).transition().ease(d3.easeLinear).duration(1500)
         .attr('cx', d => xScale(d[0]))
         .attr('cy', d => yScale(d[1]))

      
     // Remove old circles
     circles.exit().remove();

    // Define the arrow marker
    if (null){
      console.log(prevData,data)
    // Define the arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 5)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 12)
      .attr('markerHeight', 12)
      .attr('zIndex', -2)
      .attr('xoverflow', 'visible')
      .append('svg:path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#999')
      .style('stroke','none');

    // Create a mapping from id to previous data point
    const idToPrevDataPoint = new Map(prevData.map(d => [d.id, d]));

    // Filter out any data points that don't have a previous position
    const dataWithPrevPosition = data.filter(d => idToPrevDataPoint.has(d.id));

    // Draw the difference vectors
    svg.selectAll('line')
      .data(dataWithPrevPosition)
      .enter()
      .append('line')
      .attr('x1', d => idToPrevDataPoint.get(d.id).scaledX)
      .attr('y1', d => idToPrevDataPoint.get(d.id).scaledY)
      .attr('x2', d => xScale(d[0]))
      .attr('y2', d => yScale(d[1]))
      .attr('stroke', '#999')
      .attr('opacity',(d,i)=> (i%15==0)?1:0)
      .attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#arrowhead)');}


     svg.call(lasso().on("start lasso end", draw));
     draw(defaultLasso);
    


    // Bind data to text elements and add labels
/*if(labels){
  svg.selectAll('text.label')
  .data(labels)
  .enter()
  .append('text')
  .attr('class', 'label')
  .attr('fill', 'black')  
  .attr('x', d => xScale(d[0])) 
  .attr('y', d => yScale(d[1]))
  .attr('dy', '.35em')  
  .attr('text-anchor', 'middle')
  .attr('opacity', 0)
  .text((d,i) => d[2])//`Cluster ${i}`
  .attr('stroke', 'black')  
  .attr('stroke-width', '0.3px') 
  .attr('font-size', '28px')
  .attr('font-weight', '500').transition().duration(1000).attr('opacity', 1)
  ;
}*/
  

  }, [data, labels,colorCol, width, height, searchQuery,tooltipIndices]);

  return (
    <>
    <div style={{ position: 'fixed', top: '1%', left: '82%', backgroundColor: 'rgba(0, 0, 0, 0.02)',
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
        <svg ref={ref} width={width} height={height}></svg>
;
    </>
  )
}

export default Scatterplot;
//            //<div style={{ width:'150px',}}>{pieCharts}</div>
