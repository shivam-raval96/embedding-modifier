import React from 'react';
import { Pie } from 'react-chartjs-2';

const MyPieChart = () => {
  // Your data
  const dataObject = { cat: 5, bird: 11, fish: 5, elephant: 8, dog: 10 };

  // Prepare the data for the chart
  const chartData = {
    labels: Object.keys(dataObject),
    datasets: [
      {
        label: 'Animal Count',
        data: Object.values(dataObject),
        backgroundColor: [
          'rgba(255, 99, 132, 0.2)',
          'rgba(54, 162, 235, 0.2)',
          'rgba(255, 206, 86, 0.2)',
          'rgba(75, 192, 192, 0.2)',
          'rgba(153, 102, 255, 0.2)',
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  return <Pie data={chartData} />;
};

export default MyPieChart;
