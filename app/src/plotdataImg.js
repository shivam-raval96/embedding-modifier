import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { PieChart } from 'react-minimal-pie-chart';


let currentImg = ''

const r_small = 5
const r_big = 15

const colors2 = [
  "#9edae5", "#17becf", "#dbdb8d", "#bcbd22", "#c7c7c7", "#7f7f7f",
  "#f7b6d2", "#e377c2", "#c49c94", "#8c564b", "#c5b0d5", "#9467bd",
  "#ff9896", "#d62728", "#98df8a", "#2ca02c", "#ffbb78", "#ff7f0e",
  "#aec7e8", "#1f77b4"
]

const colors = [
  "#d62728", "#bcbd22", "#000000", "#17becf", "#FFD700", "#9467bd",
  "#808080", "#1f77b4", "#2ca02c", "#ff7f0e", "#7f7f7f", "#e377c2"
]
function addAlpha(color, opacity) {
  // coerce values so ti is between 0 and 1.
  var _opacity = Math.round(Math.min(Math.max(opacity || 1, 0), 1) * 255);
  return color.slice(0, 7) + _opacity.toString(16).toUpperCase();
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

function getEncodedArray(array, stringToNumberMap) {

  return array.map(str => stringToNumberMap[str]);

}







function Scatterplot(props) {
  var { width, height, data, labels, colorCol, jitter } = props;
  console.log(data)
  const margin = { top: 100, left: 120, right: 80, bottom: 100 }
  const ref = useRef();
  const [pieCharts, setPieCharts] = useState([])
  const defaultLasso = [[0, 0]]
  const [stringToNumberMap, setStringToNumberMap] = useState({ '0': 'none' });
  // const [currentImg, setCurrentImg] = useState('')

  let isZooming = false;
  let size = 30; // Default size of images
  let enlargedSize =size* 6

  data.forEach((d, i) => {
    d.id = i
    d.image = 'art/'+d[4]
  })

  useEffect(() => {    
    d3.selectAll(".autotooltip").remove();
    d3.selectAll(".tooltip").remove();


    data.forEach((d, i) => {
      d.id = i
    })



    const svg = d3.select(ref.current);
    // Define zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([1, 10])
      .on("start", () => {
        isZooming = true;
      })
      .on("zoom", (event) => {
        const currentZoom = event.transform.k; // Get current zoom level
        // Adjust the size of the images based on the zoom level
        const newSize = size / currentZoom**0.5;
        enlargedSize =newSize* 6

        svg.selectAll('g.zoomable').attr("transform", event.transform);

        svg.selectAll('image')
           .attr('width', newSize)
           .attr('height', newSize)
           .attr('x', d => xScale(d[0]) - newSize / 2) // Adjust the x position based on the new size
           .attr('y', d => yScale(d[1]) - newSize / 2); // Adjust the y position based on the new size

        
      
      })
      .on("end", () => {
        isZooming = false;
      });

    // Apply the zoom behavior to the SVG element
    svg.call(zoom);



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








    // Set scales for the scatterplot
    const xScale = d3.scaleLinear()
      .domain([d3.min(data, d => d[0]), d3.max(data, d => d[0])])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([d3.min(data, d => d[1]), d3.max(data, d => d[1])])
      .range([height - margin.bottom, margin.top]);


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




    //SetcolorCol
    if (colorCol != -1) {
      var array = getColumn(data, colorCol)
      setStringToNumberMap(encodeArray(array));



    } else {
      var array = getColumn(data, 3)
      setStringToNumberMap(encodeArray(getColumn(data, 3)))
    }



    var color_idx = getEncodedArray(array, stringToNumberMap)

    console.log(colorCol, stringToNumberMap)




    // Create a group for visual elements that should be zoomable and panable
    let zoomableGroup = svg.select('g.zoomable');
    if (zoomableGroup.empty()) {
          zoomableGroup = svg.append('g').classed('zoomable', true);
        }
    // Now, when adding your circles or images, append them to `zoomableGroup` instead of directly to `svg`
    // Bind data to images
    let images = zoomableGroup.selectAll('image')
      .data(data, d => d.id); // Use a key function for object constancy (if not already defined, ensure each data item has a unique 'id')


    // Enter new images
    images.enter().append('image')
      .attr('xlink:href', d => process.env.PUBLIC_URL + '/' + d.image) // Assuming this is correct
      .attr('width', size)
      .attr('height', size)
      .style("opacity", 0)
      .style("stroke", "red")    // set the line colour
      .attr('x', d => xScale(d[0]) - size / 2)
      .attr('y', d => yScale(d[1]) - size / 2)
      // .style("z-index", 1)
      .merge(images)
      .transition().duration(1500)
      .style("opacity", .8)
      .attr('x', d => xScale(d[0]) - size / 2)
      .attr('y', d => yScale(d[1]) - size / 2)
  


    images
      .on("mouseover", (event, d) => {
        if (isZooming) return; // Skip if zooming
        // console.log("c", currentImg);
        // console.log("d", d.image);
        // console.log(d.image === currentImg);
        if (d.image === currentImg) {
          currentImg = ''
          return
        }
        currentImg = d.image
        // console.log("e", currentImg);

        d3.select(event.currentTarget)
          .raise()
          .transition().duration(300)
          .attr('x', d => xScale(d[0]) - enlargedSize / 2) // Recalculate x to keep the image centered
          .attr('y', d => yScale(d[1]) - enlargedSize / 2) // Recalculate y to keep the image centered
          .attr('width', enlargedSize)
          .attr('height', enlargedSize)
          .style("opacity", 0.9)
          .on("end", function() {
            d3.select(this).style("pointer-events", "none");
          });
          // .style("z-index", 20);  // needs to be higher than other images
        
        console.log(d);
        //let tooltipContent = `<div><p>Title: ${d[4]}</p><p>Artist: ${d[5]}</p><p>Medium: ${d[11]}</p><p>Genre: ${d[8]}</p><p>Date/Period: ${d[10]}</p></div>`;
        let tooltipContent = `<div><p>${d[2]}</p></div>`;

        tooltip.transition()
          .duration(100)
          .style("opacity", 100);
        tooltip.html(tooltipContent)
          .style("left", (event.pageX -10 + (enlargedSize / 2)) + "px")
          .style("top", (event.pageY - 28) + "px");
      })
      .on("mouseout", (event, d) => {
        tooltip.transition()
          .duration(1500)
          .style("opacity", 0);
        d3.select(event.currentTarget)
          .transition().duration(1500)
          .attr('x', d => xScale(d[0]) - size / 2) // Reset x to original centered position
          .attr('y', d => yScale(d[1]) - size / 2) // Reset y to original centered position
          .attr('width', size)
          .attr('height', size)
          .on("end", function() { // Once the transition ends, reset pointer-events to auto
            d3.select(this).style("pointer-events", "auto");
            // console.log("out", d.image)
            // if (d.image === currentImg) {
            //   currentImg = '';
            //   console.log("cu out", currentImg)
            // }
          });
      });

  // Exit selection: Remove elements that are no longer present.
  images.exit()
    .transition().duration(1500)
    .style("opacity", 0)
    .remove();    
    // *️⃣ Disable lasso for now
    // svg.call(lasso().on("start lasso end", draw));
    // draw(defaultLasso);


    //console.log(contours)


    // Append the contours.
    var contourPaths = svg.selectAll('contours').attr("stroke-linejoin", "round")
        .attr("stroke", 'black')
        .data(images)
        .join("path")
        .style("opacity", 0) 
        .attr("fill", (d, i) => 'gray')
        .attr("stroke-width", (d, i) => i % 5 ? 0.25 : 1)
        //.attr("stroke", 'red')
        .style("z-index", -1)
        .attr("d", d3.geoPath());

    contourPaths.transition().duration(2500).style("opacity", (d, i) => i % 5 ? 0 : 0.2)





    // Bind data to text elements and add labels

    /*svg.selectAll('text.label')
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
        ;*/



  }, [data, labels, width, height, colorCol, jitter]);

  return (
    <>

      <svg ref={ref} width={width} height={height}></svg>
    </>
  )
}

export default Scatterplot;
