import React from 'react';
import styled from 'styled-components';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { BarChart3, TrendingUp, PieChart as PieIcon } from 'lucide-react';

const ChartWrapper = styled.div`
  width: 100%;
  height: 300px;
  background-color: #0d1117;
  border-radius: 8px;
  padding: 16px;
`;

const ChartHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  font-size: 14px;
  font-weight: 600;
  color: #c9d1d9;
`;

const NoChartMessage = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #7d8590;
  font-size: 14px;
  text-align: center;
`;

const COLORS = ['#58a6ff', '#238636', '#f85149', '#d29922', '#a371f7', '#7d8590'];

function ChartDisplay({ chartData, chart }) {
  // Use either chartData or chart prop
  const chartInfo = chartData || chart;
  
  if (!chartInfo) {
    return (
      <ChartWrapper>
        <NoChartMessage>
          <BarChart3 size={20} style={{ marginRight: '8px' }} />
          No chart data available
        </NoChartMessage>
      </ChartWrapper>
    );
  }

  const { type, data, title } = chartInfo;

  const renderChart = () => {
    if (!type) {
      return (
        <NoChartMessage>
          <BarChart3 size={20} style={{ marginRight: '8px' }} />
          No chart type specified
        </NoChartMessage>
      );
    }
    
    switch (type) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis 
                dataKey="name" 
                stroke="#7d8590"
                fontSize={12}
              />
              <YAxis 
                stroke="#7d8590"
                fontSize={12}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#161b22',
                  border: '1px solid #30363d',
                  borderRadius: '8px',
                  color: '#c9d1d9'
                }}
              />
              <Bar dataKey="value" fill="#58a6ff" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'line':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis 
                dataKey="name" 
                stroke="#7d8590"
                fontSize={12}
              />
              <YAxis 
                stroke="#7d8590"
                fontSize={12}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#161b22',
                  border: '1px solid #30363d',
                  borderRadius: '8px',
                  color: '#c9d1d9'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#58a6ff" 
                strokeWidth={2}
                dot={{ fill: '#58a6ff', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'pie':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#161b22',
                  border: '1px solid #30363d',
                  borderRadius: '8px',
                  color: '#c9d1d9'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        );

      case 'area':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis 
                dataKey="name" 
                stroke="#7d8590"
                fontSize={12}
              />
              <YAxis 
                stroke="#7d8590"
                fontSize={12}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#161b22',
                  border: '1px solid #30363d',
                  borderRadius: '8px',
                  color: '#c9d1d9'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="#58a6ff" 
                fill="#58a6ff" 
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        );

      default:
        return (
          <NoChartMessage>
            <BarChart3 size={20} style={{ marginRight: '8px' }} />
            Unsupported chart type: {type}
          </NoChartMessage>
        );
    }
  };

  const getChartIcon = () => {
    switch (type) {
      case 'bar':
        return <BarChart3 size={16} />;
      case 'line':
        return <TrendingUp size={16} />;
      case 'pie':
        return <PieIcon size={16} />;
      default:
        return <BarChart3 size={16} />;
    }
  };

  return (
    <ChartWrapper>
      <ChartHeader>
        {getChartIcon()}
        {title || `${type ? type.charAt(0).toUpperCase() + type.slice(1) : 'Unknown'} Chart`}
      </ChartHeader>
      {renderChart()}
    </ChartWrapper>
  );
}

export default ChartDisplay; 