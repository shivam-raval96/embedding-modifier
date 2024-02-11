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

const colors = [
    "#d62728", "#bcbd22","#000000","#17becf", "#FFD700","#9467bd",
     "#808080","#1f77b4", "#2ca02c", "#ff7f0e", "#7f7f7f","#e377c2"
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
  var { width, height, data , labels, colorCol, jitter} = props;
  const margin = {top:100, left:120, right:80, bottom:100}
  const ref = useRef();
  const [pieCharts, setPieCharts] = useState([])
  const defaultLasso = [[0,0]]

  useEffect(() => {
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


    // Bind data to circles and add tooltips
    //console.log(labels)
    // Compute the density contours.
    const contours = d3.contourDensity()
    .x(d => xScale(d[0]))
    .y(d => yScale(d[1]))
    .size([width, height])
    .bandwidth(40)
    .thresholds(3)
    (data);



    //console.log(contours)
    

    // Append the contours.
    /*var contourPaths = svg.selectAll('contours').attr("stroke-linejoin", "round")
        .attr("stroke", 'black')
        .data(contours)
        .join("path")
        .style("opacity", 0) 
        .attr("fill", (d, i) => 'gray')
        .attr("stroke-width", (d, i) => i % 5 ? 0.25 : 1)
        //.attr("stroke", 'red')
        .style("z-index", -1)
        .attr("d", d3.geoPath());

    contourPaths.transition().duration(2500).style("opacity", (d, i) => i % 5 ? 0 : 0.2)*/ 

    //SetcolorCol
    if (colorCol!=-1){
        var array = getColumn(data,colorCol)
        var stringToNumberMap = encodeArray(array);


    }else{
        var array = getColumn(data,3)
        var stringToNumberMap= encodeArray(getColumn(data,3))

    }



    var color_idx = getEncodedArray(array, stringToNumberMap)

    let result = getPieDistribution(getColumn(data,colorCol), getColumn(data,3));

    console.log(colorCol,stringToNumberMap)

     // Bind data to circles
     const circles = svg.selectAll('circle').data(data);

     // Enter new circles
     circles.enter().append('circle')
         .attr('cx', d => xScale(d[0]))
         .attr('cy', d => yScale(d[1]))
         .attr('r', r_small)
        .style("opacity", 0.9)
        .attr('fill', (d,i) =>  colors[color_idx[i]%10])
        .attr('stroke', 'black')  // Add this line for the boundary color
        .attr('stroke-width', 0.5)  // Add this line for the boundary width
        .style("z-index", 2)
        .on("mouseover", (event, d) => {

            svg.selectAll('circle')
            .transition().duration(100)
            .style("opacity", 0.9) 
            .attr('stroke-width', 0.5) 
            .attr('r', r_small);
            d3.select(event.currentTarget).transition().duration(100)
                .style("opacity", 1) 
                .attr('stroke-width', 0.5)  // Add this line for the boundary width
                .attr('r', r_big);



            tooltip.transition()
                .duration(100)
                .style("opacity", .9);
            tooltip.html(d[2])
                .style("left", (event.pageX + 5) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", (d) => {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        })

     // Update existing circles
     circles.attr('fill', (d,i) =>  colors[color_idx[i]%10]).transition().ease(d3.easeLinear).duration(1500)
         .attr('cx', d => xScale(d[0]))
         .attr('cy', d => yScale(d[1]));

     // Remove old circles
     circles.exit().remove();

     svg.call(lasso().on("start lasso end", draw));
     draw(defaultLasso);
    
    /*var points=svg.selectAll('circle')
        .data(data)
        .enter()
        .append('circle')
        .attr('cx', d => xScale(d[0]))
        .attr('cy', d => yScale(d[1]))
        .attr('r', 4)
        .style("opacity", 0.9)
        .attr('fill', (d,i) =>  colors[color_idx[i]%10])
        .attr('stroke', 'black')  // Add this line for the boundary color
        .attr('stroke-width', 0.5)  // Add this line for the boundary width
        .on("mouseover", (event, d) => {

            svg.selectAll('circle')
            .transition().duration(100)
            .style("opacity", 0.9) 
            .attr('stroke-width', 0.5) 
            .attr('r', 4);
            d3.select(event.currentTarget).transition().duration(100)
                .style("opacity", 1) 
                .attr('stroke-width', 0.5)  // Add this line for the boundary width
                .attr('r', 10);



            tooltip.transition()
                .duration(100)
                .style("opacity", .9);
            tooltip.html(d[2])
                .style("left", (event.pageX + 5) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", (d) => {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        }).transition() // start a transition
        .duration(2000) // 2 seconds;

        
        
        if (jitter){
            points.attr("cx", function(d,i) { return xScale(d[0])+ 2*Math.random() });
            points.attr("cy", function(d,i) { return yScale(d[1])+ 2*Math.random() });
            
        }*/



    // Bind data to text elements and add labels

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
        .text((d,i) => `Cluster ${i}`)
        .attr('stroke', 'black')  
        .attr('stroke-width', '0.3px') 
        .attr('font-size', '28px')
        .attr('font-weight', '500').transition().duration(1000).attr('opacity', 1)
        ;
    
        var labeldata_all = []
        for (const [label, value] of Object.entries(result)) {
            var piedata = []
            var i =0
            for (const [key, count] of Object.entries(value)){
                
                piedata.push({ title: key, value: count, color: colors[stringToNumberMap[key]] })
                i+=1
            }
            labeldata_all.push(piedata)
          }


   
        
        var pies = []
        labeldata_all.forEach((e,i)=>{
            pies.push(<><p style={{position:'relative', top:'40px',left:'20px'}}>Cluster {i}</p><PieChart data={e} label={({ dataEntry }) => dataEntry.title}
        labelStyle={(index) => ({
            fontSize: '8px',
            fontFamily: 'sans-serif',
          })}
          center={[65,50]}
          radius={20}
          labelPosition={120}/></>)
        })

        setPieCharts(pies)
        console.log(array)

  }, [data, labels, width, height, colorCol, jitter]);

  return (
    <>
            
        <svg ref={ref} width={width} height={height}></svg>
        <div id = 'legend' style={{ position: 'fixed', top: '2%', left: '86%', width: '200px',height:'95%', backgroundColor: 'rgba(0, 0, 0, 0.02)',
                 boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.5)',  // Drop shadow
                borderRadius: '20px' ,                         // Curved edges
                fontFamily: 'Perpetua',  // Setting the font family
                overflowY: 'scroll'
                }}>
        
            <div style={{ width:'150px',}}>{pieCharts}</div>
            
            </div>;
    </>
  )
}

export default Scatterplot;
