import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';

function ScatterPlot() {
    const [data, setData] = useState(generateRandomData());
    const svgRef = useRef();

    useEffect(() => {
        drawScatterPlot();
    }, [data]);

    function generateRandomData() {
        // Generate random data points
        return Array.from({ length: 50 }, () => ({
            x: Math.random() * 100,
            y: Math.random() * 100
        }));
    }

    function drawScatterPlot() {
        const svg = d3.select(svgRef.current);
        const width = +svg.attr('width');
        const height = +svg.attr('height');

        // Scale functions
        const xScale = d3.scaleLinear().domain([-5, 105]).range([0, width-20]);
        const yScale = d3.scaleLinear().domain([-5, 105]).range([height-30, 0]);

        // Bind data to circles
        const circles = svg.selectAll('circle').data(data);

        // Enter new circles
        circles.enter().append('circle')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 5)
            .style('fill', 'lightsteelblue')
            .style('stroke-width', '0.5')
            .style('stroke', 'black');
            ;

        // Update existing circles
        circles.transition().duration(2000)
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y));

        // Remove old circles
        circles.exit().remove();
    }

    function updateData() {
        setData(generateRandomData());
    }

    return (
        <div>
            <svg ref={svgRef} width={400} height={400}></svg>
            <button onClick={updateData}>Move Points</button>
        </div>
    );
}

export default ScatterPlot;
