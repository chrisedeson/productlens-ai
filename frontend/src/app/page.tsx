'use client';

import { useState, useEffect } from 'react';
import { TextSearch } from '@/components/TextSearch';
import { OCRSearch } from '@/components/OCRSearch';
import { ImageSearch } from '@/components/ImageSearch';
import { checkHealth } from '@/lib/api';

type TabType = 'text' | 'ocr' | 'image';

interface ServiceStatus {
  recommendation: boolean;
  ocr: boolean;
  cnn: boolean;
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>('text');
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(null);
  const [isConnected, setIsConnected] = useState<boolean | null>(null);

  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const health = await checkHealth();
        setServiceStatus(health.services);
        setIsConnected(true);
      } catch {
        setIsConnected(false);
      }
    };

    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const tabs = [
    {
      id: 'text' as TabType,
      name: 'Text Search',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      ),
      description: 'Natural language product search',
      color: 'blue',
      available: serviceStatus?.recommendation ?? true,
    },
    {
      id: 'ocr' as TabType,
      name: 'Handwritten OCR',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
        </svg>
      ),
      description: 'Extract text from handwritten notes',
      color: 'purple',
      available: serviceStatus?.ocr ?? true,
    },
    {
      id: 'image' as TabType,
      name: 'Image Recognition',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      ),
      description: 'Identify products from images',
      color: 'emerald',
      available: serviceStatus?.cnn ?? true,
    },
  ];

  const getTabStyles = (tab: typeof tabs[0], isActive: boolean) => {
    const colors = {
      blue: isActive
        ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30'
        : 'text-slate-600 dark:text-slate-400 hover:bg-blue-50 dark:hover:bg-blue-900/30',
      purple: isActive
        ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/30'
        : 'text-slate-600 dark:text-slate-400 hover:bg-purple-50 dark:hover:bg-purple-900/30',
      emerald: isActive
        ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-500/30'
        : 'text-slate-600 dark:text-slate-400 hover:bg-emerald-50 dark:hover:bg-emerald-900/30',
    };
    return colors[tab.color as keyof typeof colors];
  };

  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 shadow-sm border-b border-slate-200 dark:border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
                  ProductLens AI
                </h1>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Intelligent E-commerce Product Recommendations
                </p>
              </div>
            </div>
            
            {/* Connection Status */}
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${
                isConnected === null
                  ? 'bg-yellow-400 animate-pulse'
                  : isConnected
                    ? 'bg-green-400'
                    : 'bg-red-400'
              }`} />
              <span className="text-sm text-slate-500 dark:text-slate-400">
                {isConnected === null
                  ? 'Connecting...'
                  : isConnected
                    ? 'Connected'
                    : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="flex flex-wrap gap-3 mb-8 justify-center">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              disabled={!tab.available}
              className={`flex items-center gap-3 px-6 py-3 rounded-xl font-medium transition-all duration-200 ${
                getTabStyles(tab, activeTab === tab.id)
              } ${!tab.available ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {tab.icon}
              <div className="text-left">
                <div className="font-semibold">{tab.name}</div>
                <div className={`text-xs ${
                  activeTab === tab.id ? 'text-white/80' : 'text-slate-400'
                }`}>
                  {tab.description}
                </div>
              </div>
              {!tab.available && (
                <span className="ml-2 text-xs bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-400 px-2 py-0.5 rounded">
                  Unavailable
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-8 border border-slate-200 dark:border-slate-700">
          {activeTab === 'text' && <TextSearch />}
          {activeTab === 'ocr' && <OCRSearch />}
          {activeTab === 'image' && <ImageSearch />}
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-sm text-slate-500 dark:text-slate-400">
          <p>Built with Next.js, TensorFlow, Pinecone, and OpenAI</p>
          <p className="mt-1">
            Data Science Coding Task • 2025
          </p>
        </footer>
      </div>
    </main>
  );
}
