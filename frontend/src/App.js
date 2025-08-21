import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { Send, Upload, BarChart3, FileText, Bot, User, Download, Brain, ArrowLeft, Plus, Pencil, Trash2, RefreshCw, Settings, Paperclip, Mic, Zap } from 'lucide-react';
import FileUpload from './components/FileUpload';
import ChatMessage from './components/ChatMessage';
import DataPreview from './components/DataPreview';
import ChartDisplay from './components/ChartDisplay';
import QuickMenu from './components/QuickMenu';

const AppContainer = styled.div`
  display: flex;
  height: 100vh;
  background-color: #0d1117;
  color: #c9d1d9;
`;

const Sidebar = styled.div`
  width: 280px;
  background-color: #161b22;
  border-right: 1px solid #30363d;
  display: flex;
  flex-direction: column;
  padding: 20px;
`;

const MainContent = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: #0d1117;
`;

const Header = styled.div`
  padding: 20px 30px;
  border-bottom: 1px solid #30363d;
  background-color: #161b22;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const HeaderLeft = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const HeaderRight = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const BrandSection = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const BrandIcon = styled(Brain)`
  color: #a855f7;
`;

const BrandText = styled.div`
  font-size: 18px;
  font-weight: 600;
  color: #a855f7;
`;

const WelcomeText = styled.div`
  font-size: 14px;
  color: #8b949e;
  margin-top: 2px;
`;

const AgentStatus = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #238636;
`;

const StatusDot = styled.div`
  width: 8px;
  height: 8px;
  background-color: #238636;
  border-radius: 50%;
`;

const HeaderButton = styled.button`
  background: none;
  border: none;
  color: #8b949e;
  cursor: pointer;
  padding: 8px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;

  &:hover {
    background-color: #21262d;
    color: #c9d1d9;
  }
`;

const ChatContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 30px;
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const InputContainer = styled.div`
  padding: 20px 30px;
  border-top: 1px solid #30363d;
  background-color: #161b22;
`;

const InputWrapper = styled.div`
  display: flex;
  gap: 12px;
  align-items: flex-end;
  max-width: 100%;
  margin: 0 auto;
  background-color: #0d1117;
  border: 1px solid #30363d;
  border-radius: 12px;
  padding: 12px;
`;

const TextArea = styled.textarea`
  flex: 1;
  min-height: 24px;
  max-height: 200px;
  padding: 8px 12px;
  border: none;
  border-radius: 8px;
  background-color: transparent;
  color: #c9d1d9;
  font-size: 14px;
  resize: none;
  outline: none;
  font-family: inherit;
  
  &::placeholder {
    color: #7d8590;
  }
`;

const InputButton = styled.button`
  background: none;
  border: none;
  color: #8b949e;
  cursor: pointer;
  padding: 8px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;

  &:hover {
    background-color: #21262d;
    color: #c9d1d9;
  }

  &.send {
    background-color: #a855f7;
    color: white;
    
    &:hover {
      background-color: #9333ea;
    }
  }
`;

const SidebarTitle = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: #c9d1d9;
  margin-bottom: 20px;
`;

const NewChatButton = styled.button`
  width: 100%;
  padding: 12px 16px;
  background-color: #a855f7;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-bottom: 20px;
  transition: background-color 0.2s;

  &:hover {
    background-color: #9333ea;
  }
`;

const ChatSession = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  background-color: ${props => props.active ? '#21262d' : 'transparent'};
  border-left: 3px solid ${props => props.active ? '#a855f7' : 'transparent'};
  border-radius: 6px;
  cursor: pointer;
  margin-bottom: 8px;
  transition: all 0.2s;

  &:hover {
    background-color: #21262d;
  }
`;

const ChatInfo = styled.div`
  flex: 1;
`;

const ChatName = styled.div`
  font-size: 14px;
  color: #c9d1d9;
  margin-bottom: 2px;
`;

const ChatMeta = styled.div`
  font-size: 12px;
  color: #8b949e;
`;

const ChatActions = styled.div`
  display: flex;
  gap: 4px;
  opacity: 0;
  transition: opacity 0.2s;

  ${ChatSession}:hover & {
    opacity: 1;
  }
`;

const ActionButton = styled.button`
  background: none;
  border: none;
  color: #8b949e;
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;

  &:hover {
    background-color: #30363d;
    color: #c9d1d9;
  }
`;

const MainTitle = styled.h1`
  font-size: 32px;
  font-weight: 700;
  color: #a855f7;
  text-align: center;
  margin-bottom: 8px;
`;

const MainSubtitle = styled.p`
  font-size: 16px;
  color: #8b949e;
  text-align: center;
  margin-bottom: 40px;
`;

const InsightCards = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 40px;
`;

const InsightCard = styled.div`
  background-color: #161b22;
  border: 1px solid #30363d;
  border-radius: 12px;
  padding: 24px;
`;

const CardHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
`;

const CardIcon = styled.div`
  width: 40px;
  height: 40px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${props => props.color};
  color: white;
`;

const CardTitle = styled.h3`
  font-size: 18px;
  font-weight: 600;
  color: #c9d1d9;
  margin: 0;
`;

const CardSubtitle = styled.p`
  font-size: 14px;
  color: #8b949e;
  margin: 0;
`;

const QueryList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const QueryItem = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 14px;
  color: #c9d1d9;

  &:hover {
    background-color: #21262d;
  }
`;

const QueryIcon = styled.div`
  color: ${props => props.color};
`;

const InstructionText = styled.p`
  text-align: center;
  color: #8b949e;
  font-size: 14px;
  margin-top: 20px;
`;

const WelcomeMessage = styled.div`
  text-align: center;
  padding: 40px 20px;
  color: #7d8590;
  
  h2 {
    font-size: 24px;
    margin-bottom: 10px;
    color: #c9d1d9;
  }
  
  p {
    font-size: 14px;
    line-height: 1.5;
  }
`;

const SidebarSection = styled.div`
  margin-bottom: 30px;
`;

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [dataPreview, setDataPreview] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Initialize session and upload manufacturing data
  useEffect(() => {
    const initializeSession = async () => {
      try {
        const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
        
        // Create session
        const sessionResponse = await fetch(`${apiUrl}/session/create`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });
        
        if (!sessionResponse.ok) {
          throw new Error('Failed to create session');
        }
        
        const sessionData = await sessionResponse.json();
        const newSessionId = sessionData.session_id;
        setSessionId(newSessionId);
        
        // Upload manufacturing data automatically
        const response = await fetch('/sample_manufacturing_data.csv');
        const csvText = await response.text();
        const blob = new Blob([csvText], { type: 'text/csv' });
        const file = new File([blob], 'sample_manufacturing_data.csv', { type: 'text/csv' });
        
        const formData = new FormData();
        formData.append('file', file);
        
        const uploadResponse = await fetch(`${apiUrl}/session/${newSessionId}/upload`, {
          method: 'POST',
          body: formData
        });
        
        if (uploadResponse.ok) {
          const uploadData = await uploadResponse.json();
          setUploadedFile({ name: 'sample_manufacturing_data.csv' });
          
          // Use the preview data from the backend
          const previewData = uploadData.dataset_info?.preview || null;
          setDataPreview(previewData);
          
          // Add welcome message
          const welcomeMessage = {
            id: Date.now(),
            type: 'bot',
            content: '🏭 Manufacturing data loaded! I can help you analyze production lines, quality scores, downtime, and efficiency metrics. What would you like to explore?',
            timestamp: new Date()
          };
          setMessages([welcomeMessage]);
        }
        
      } catch (error) {
        console.error('Failed to initialize session:', error);
        const errorMessage = {
          id: Date.now(),
          type: 'bot',
          content: '⚠️ Failed to load manufacturing data. You can upload your own data file to get started.',
          timestamp: new Date()
        };
        setMessages([errorMessage]);
      } finally {
        setIsInitializing(false);
      }
    };
    
    initializeSession();
  }, []);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading || !sessionId) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          query: inputValue
        })
      });

      const data = await response.json();
      
      // Debug: Log what we're receiving from backend
      console.log('Backend response data:', data);
      
      if (data.success) {
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: data.response,
          chart: data.artifacts?.chart_data ? {
            type: data.artifacts.chart_type || 'bar',
            data: data.artifacts.chart_data,
            title: 'Analysis Chart'
          } : null,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        throw new Error(data.error || 'Analysis failed');
      }
      
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Sorry, I encountered an error while processing your request. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleFileUpload = async (file) => {
    setUploadedFile(file);
    setIsUploading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const response = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (response.ok) {
        setDataPreview(result.preview);
        // Add a success message
        const successMessage = {
          id: Date.now(),
          type: 'bot',
          content: `✅ File "${file.name}" uploaded successfully! You can now ask questions about your data.`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, successMessage]);
      } else {
        console.error('Upload failed:', result.error);
        const errorMessage = {
          id: Date.now(),
          type: 'bot',
          content: `❌ Upload failed: ${result.error}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
        
        // Fallback to local preview
        const reader = new FileReader();
        reader.onload = (e) => {
          const content = e.target.result;
          const lines = content.split('\n').slice(0, 3);
          setDataPreview({
            columns: ['Content'],
            rows: lines.map(line => [line.substring(0, 50)])
          });
        };
        reader.readAsText(file);
      }
    } catch (error) {
      console.error('Upload error:', error);
      const errorMessage = {
        id: Date.now(),
        type: 'bot',
        content: `❌ Upload error: ${error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      
      // Fallback to local preview
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        const lines = content.split('\n').slice(0, 3);
        setDataPreview({
          columns: ['Content'],
          rows: lines.map(line => [line.substring(0, 50)])
        });
      };
      reader.readAsText(file);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRemoveFile = () => {
    setUploadedFile(null);
    setDataPreview(null);
  };

  const handleQuickMenuSelect = (query) => {
    setInputValue(query);
  };

  const handleQueryClick = (query) => {
    setInputValue(query);
  };

  const handlePaperclipClick = () => {
    // Trigger file input click
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.csv,.xlsx,.xls,.txt';
    fileInput.onchange = (e) => {
      if (e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
      }
    };
    fileInput.click();
  };

  const handleClearChat = () => {
    setMessages([]);
    setInputValue('');
  };

  const handleNewChat = () => {
    setMessages([]);
    setInputValue('');
    setUploadedFile(null);
    setDataPreview(null);
  };

  const handleRefresh = () => {
    // Reload the page to refresh everything
    window.location.reload();
  };

  const handleMicClick = () => {
    // For now, just show an alert. In the future, this could integrate with speech-to-text
    alert('Voice input feature coming soon! For now, please type your query.');
  };

  const handleEditChat = () => {
    // For now, just show an alert. In the future, this could allow editing chat name
    alert('Edit chat name feature coming soon!');
  };

  const handleDeleteChat = () => {
    if (window.confirm('Are you sure you want to delete this chat?')) {
      handleNewChat();
    }
  };

  const manufacturingQueries = [
    {
      category: 'Production Analytics',
      icon: <BarChart3 size={20} />,
      iconColor: '#3b82f6',
      subtitle: 'BOM comparison, defects, costs',
      queries: [
        'Compare production vs. BOM quantities',
        'Show defective production rates by part',
        'Analyze cost per unit trends',
        'Production summary for last quarter',
        'Top 5 most produced parts this month',
        'Show production efficiency metrics'
      ]
    },
    {
      category: 'Utilization Reports',
      icon: <Zap size={20} />,
      iconColor: '#10b981',
      subtitle: 'Capacity, efficiency, bottlenecks',
      queries: [
        'Show plant utilization by location',
        'Top reasons for capacity leakage',
        'Monthly utilization trends',
        'Compare rated vs actual capacity',
        'Plant performance analysis',
        'Bottleneck identification report'
      ]
    }
  ];

  // Show loading while initializing
  if (isInitializing) {
    return (
      <AppContainer>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh', 
          flexDirection: 'column',
          gap: '20px'
        }}>
          <div style={{ fontSize: '24px' }}>🏭</div>
          <div>Loading Manufacturing Data...</div>
        </div>
      </AppContainer>
    );
  }

  return (
    <AppContainer>
      <Sidebar>
        <SidebarTitle>
          <ArrowLeft size={16} />
          Chat Sessions
        </SidebarTitle>

        <NewChatButton onClick={handleNewChat}>
          <Plus size={16} />
          New Chat
        </NewChatButton>

        <ChatSession active>
          <ChatInfo>
            <ChatName>New Chat</ChatName>
            <ChatMeta>{messages.length} messages • Just now</ChatMeta>
          </ChatInfo>
          <ChatActions>
            <ActionButton onClick={handleEditChat}>
              <Pencil size={12} />
            </ActionButton>
            <ActionButton onClick={handleDeleteChat}>
              <Trash2 size={12} />
            </ActionButton>
          </ChatActions>
        </ChatSession>

        <SidebarSection>
          <FileUpload
            onFileUpload={handleFileUpload}
            uploadedFile={uploadedFile}
            onRemoveFile={handleRemoveFile}
          />
        </SidebarSection>

        {dataPreview && (
          <SidebarSection>
            <DataPreview data={dataPreview} />
          </SidebarSection>
        )}
      </Sidebar>

      <MainContent>
        <Header>
          <HeaderLeft>
            <BrandSection>
              <BrandIcon size={24} />
              <div>
                <BrandText>ManuVerse.AI</BrandText>
                <WelcomeText>Welcome, jane_engineer (engineer)</WelcomeText>
              </div>
            </BrandSection>
          </HeaderLeft>
          
          <HeaderRight>
            <AgentStatus>
              <StatusDot />
              Agent
            </AgentStatus>
            <HeaderButton onClick={handleRefresh}>
              <RefreshCw size={16} />
            </HeaderButton>
            <HeaderButton onClick={handleClearChat}>Clear Chat</HeaderButton>
            <HeaderButton>
              <Settings size={16} />
            </HeaderButton>
          </HeaderRight>
        </Header>

        <ChatContainer ref={chatContainerRef}>
          {messages.length === 0 ? (
            <>
              <MainTitle>Manufacturing Intelligence</MainTitle>
              <MainSubtitle>Get instant insights from your production and utilization data</MainSubtitle>
              
              <InsightCards>
                {manufacturingQueries.map((category, index) => (
                  <InsightCard key={index}>
                    <CardHeader>
                      <CardIcon color={category.iconColor}>
                        {category.icon}
                      </CardIcon>
                      <div>
                        <CardTitle>{category.category}</CardTitle>
                        <CardSubtitle>{category.subtitle}</CardSubtitle>
                      </div>
                    </CardHeader>
                    
                    <QueryList>
                      {category.queries.map((query, queryIndex) => (
                        <QueryItem 
                          key={queryIndex}
                          onClick={() => handleQueryClick(query)}
                        >
                          <QueryIcon color={category.iconColor}>
                            {category.category === 'Production Analytics' ? 
                              <BarChart3 size={14} /> : <Zap size={14} />
                            }
                          </QueryIcon>
                          {query}
                        </QueryItem>
                      ))}
                    </QueryList>
                  </InsightCard>
                ))}
              </InsightCards>
              
              <InstructionText>
                Click any question above or type your own manufacturing query below
              </InstructionText>
            </>
          ) : (
            messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))
          )}
          
          {isLoading && (
            <ChatMessage 
              message={{
                id: 'loading',
                type: 'bot',
                content: 'Analyzing your data...',
                timestamp: new Date()
              }}
              isLoading={true}
            />
          )}
          
          {isUploading && (
            <ChatMessage 
              message={{
                id: 'uploading',
                type: 'bot',
                content: 'Uploading your file...',
                timestamp: new Date()
              }}
              isLoading={true}
            />
          )}
        </ChatContainer>

        <InputContainer>
          <InputWrapper>
            <InputButton onClick={handlePaperclipClick}>
              <Paperclip size={16} />
            </InputButton>
            <TextArea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about production, utilization, or upload CSV/Excel files for analysis..."
              rows={1}
            />
            <InputButton onClick={handleMicClick}>
              <Mic size={16} />
            </InputButton>
            <InputButton 
              className="send"
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
            >
              <Send size={16} />
            </InputButton>
          </InputWrapper>
        </InputContainer>
      </MainContent>
    </AppContainer>
  );
}

export default App; 