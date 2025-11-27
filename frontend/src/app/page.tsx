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
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const tabs = [
    {
      id: 'text' as TabType,
      name: 'Text Search',
      shortName: 'Text',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      ),
      description: 'Search with natural language',
      gradient: 'from-indigo-500 to-purple-500',
      bgActive: 'bg-indigo-500',
      bgHover: 'hover:bg-indigo-50 dark:hover:bg-indigo-950/30',
      available: serviceStatus?.recommendation ?? true,
    },
    {
      id: 'ocr' as TabType,
      name: 'Handwriting OCR',
      shortName: 'OCR',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
        </svg>
      ),
      description: 'Upload handwritten notes',
      gradient: 'from-violet-500 to-fuchsia-500',
      bgActive: 'bg-violet-500',
      bgHover: 'hover:bg-violet-50 dark:hover:bg-violet-950/30',
      available: serviceStatus?.ocr ?? true,
    },
    {
      id: 'image' as TabType,
      name: 'Image Recognition',
      shortName: 'Image',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      ),
      description: 'Identify products from photos',
      gradient: 'from-emerald-500 to-teal-500',
      bgActive: 'bg-emerald-500',
      bgHover: 'hover:bg-emerald-50 dark:hover:bg-emerald-950/30',
      available: serviceStatus?.cnn ?? true,
    },
  ];

  const activeTabData = tabs.find(t => t.id === activeTab);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      {/* Hero Header */}
      <header className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-pink-500/10" />
        <div className="absolute inset-0">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-indigo-400/20 rounded-full blur-3xl" />
          <div className="absolute top-0 right-1/4 w-96 h-96 bg-purple-400/20 rounded-full blur-3xl" />
        </div>
        
        <div className="relative container-responsive py-8 sm:py-12">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl blur-lg opacity-50" />
                <div className="relative w-14 h-14 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-xl">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
              </div>
              <div>
                <h1 className="text-2xl sm:text-3xl font-bold gradient-text">
                  ProductLens AI
                </h1>
                <p className="text-sm sm:text-base text-slate-500 dark:text-slate-400">
                  Smart Product Recommendations
                </p>
              </div>
            </div>
            
            {/* Connection Status */}
            <div className="flex items-center gap-3 px-4 py-2 rounded-full bg-white/80 dark:bg-slate-800/80 backdrop-blur shadow-sm border border-slate-200 dark:border-slate-700">
              <div className={`w-2.5 h-2.5 rounded-full transition-colors ${
                isConnected === null
                  ? 'bg-amber-400 animate-pulse'
                  : isConnected
                    ? 'bg-emerald-400'
                    : 'bg-red-400'
              }`} />
              <span className="text-sm font-medium text-slate-600 dark:text-slate-300">
                {isConnected === null
                  ? 'Connecting...'
                  : isConnected
                    ? 'API Connected'
                    : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="container-responsive pb-12">
        {/* Tab Navigation - Card Style */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 mb-8 mt-6 relative z-10">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                disabled={!tab.available}
                className={`group relative overflow-hidden rounded-2xl p-4 sm:p-5 text-left transition-all duration-300 ${
                  isActive
                    ? `bg-gradient-to-br ${tab.gradient} text-white shadow-lg scale-[1.02]`
                    : `bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 ${tab.bgHover} shadow-sm`
                } ${!tab.available ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
              >
                {isActive && (
                  <div className="absolute inset-0 bg-white/10" />
                )}
                <div className="relative flex items-start gap-3">
                  <div className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center ${
                    isActive 
                      ? 'bg-white/20' 
                      : 'bg-slate-100 dark:bg-slate-700'
                  }`}>
                    <span className={isActive ? 'text-white' : 'text-slate-600 dark:text-slate-300'}>
                      {tab.icon}
                    </span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className={`font-semibold ${
                      isActive ? 'text-white' : 'text-slate-800 dark:text-slate-100'
                    }`}>
                      <span className="hidden sm:inline">{tab.name}</span>
                      <span className="sm:hidden">{tab.shortName}</span>
                    </div>
                    <div className={`text-sm mt-0.5 ${
                      isActive ? 'text-white/80' : 'text-slate-500 dark:text-slate-400'
                    }`}>
                      {tab.description}
                    </div>
                  </div>
                </div>
                {!tab.available && (
                  <div className="absolute top-2 right-2 px-2 py-0.5 bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-400 text-xs font-medium rounded-full">
                    Offline
                  </div>
                )}
              </button>
            );
          })}
        </div>

        {/* Main Content Card */}
        <div className="bg-white dark:bg-slate-800 rounded-3xl shadow-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          {/* Tab Header Bar */}
          <div className={`px-6 py-4 bg-gradient-to-r ${activeTabData?.gradient} flex items-center gap-3`}>
            <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center text-white">
              {activeTabData?.icon}
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">{activeTabData?.name}</h2>
              <p className="text-sm text-white/80">{activeTabData?.description}</p>
            </div>
          </div>
          
          {/* Tab Content */}
          <div className="p-6 sm:p-8">
            <div className="animate-fade-in">
              {activeTab === 'text' && <TextSearch />}
              {activeTab === 'ocr' && <OCRSearch />}
              {activeTab === 'image' && <ImageSearch />}
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-100 dark:bg-slate-800 text-sm text-slate-500 dark:text-slate-400">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Powered by TensorFlow, Pinecone & OpenAI
          </div>
          <p className="mt-3 text-xs text-slate-400 dark:text-slate-500">
            Data Science Coding Task • 2025
          </p>
        </footer>
      </main>
    </div>
  );
}
