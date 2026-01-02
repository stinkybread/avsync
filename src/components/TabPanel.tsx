import { ReactNode, useState } from 'react'
import './TabPanel.css'

interface Tab {
  id: string
  label: string
  content: ReactNode
}

interface TabPanelProps {
  tabs: Tab[]
  defaultTab?: string
  theme: 'dark' | 'light'
  onToggleTheme: () => void
  activeTab?: string
  onTabChange?: (tabId: string) => void
}

export default function TabPanel({ tabs, defaultTab, theme, onToggleTheme, activeTab: controlledActiveTab, onTabChange }: TabPanelProps) {
  const [internalActiveTab, setInternalActiveTab] = useState(defaultTab || tabs[0]?.id)

  const activeTab = controlledActiveTab !== undefined ? controlledActiveTab : internalActiveTab

  const handleTabChange = (tabId: string) => {
    if (onTabChange) {
      onTabChange(tabId)
    } else {
      setInternalActiveTab(tabId)
    }
  }

  const activeTabContent = tabs.find(tab => tab.id === activeTab)?.content

  return (
    <div className="tab-panel">
      <div className="tab-panel-header">
        <div className="app-branding">
          <img src="/logo.svg" alt="AVSync Logo" className="app-logo" width="32" height="32" style={{ filter: 'invert(77%) sepia(82%) saturate(439%) hue-rotate(359deg) brightness(98%) contrast(98%)' }} />
          <div className="app-branding-text">
            <h1 className="app-title">
              AVSync <span className="app-subtitle">Automated Audio Video Synchronization</span>
            </h1>
          </div>
        </div>
        <button className="theme-toggle" onClick={onToggleTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
          {theme === 'dark' ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="5"/>
              <line x1="12" y1="1" x2="12" y2="3"/>
              <line x1="12" y1="21" x2="12" y2="23"/>
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
              <line x1="1" y1="12" x2="3" y2="12"/>
              <line x1="21" y1="12" x2="23" y2="12"/>
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
          )}
        </button>
      </div>
      <div className="tab-header">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => handleTabChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="tab-content">
        {activeTabContent}
      </div>
    </div>
  )
}
