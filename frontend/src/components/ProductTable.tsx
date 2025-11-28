'use client';

import { Product } from '@/lib/api';

interface ProductTableProps {
  products: Product[];
  title?: string;
}

export function ProductTable({ products, title }: ProductTableProps) {
  if (products.length === 0) {
    return null;
  }

  return (
    <div className="mt-8 animate-fade-in">
      {title && (
        <div className="flex items-center gap-3 mb-5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
            </svg>
          </div>
          <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            {title}
            <span className="ml-2 text-sm font-normal text-slate-400">({products.length})</span>
          </h2>
        </div>
      )}
      
      <div className="overflow-x-auto rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm">
        <table className="w-full border-collapse bg-white dark:bg-slate-800">
          <thead>
            <tr className="bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-700 dark:to-slate-750">
              <th className="px-6 py-4 text-left text-xs font-bold text-slate-600 dark:text-slate-300 uppercase tracking-wider border-b border-slate-200 dark:border-slate-600">
                Stock Code
              </th>
              <th className="px-6 py-4 text-left text-xs font-bold text-slate-600 dark:text-slate-300 uppercase tracking-wider border-b border-slate-200 dark:border-slate-600">
                Description
              </th>
              <th className="px-6 py-4 text-left text-xs font-bold text-slate-600 dark:text-slate-300 uppercase tracking-wider border-b border-slate-200 dark:border-slate-600">
                Unit Price
              </th>
              <th className="px-6 py-4 text-left text-xs font-bold text-slate-600 dark:text-slate-300 uppercase tracking-wider border-b border-slate-200 dark:border-slate-600">
                Country
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
            {products.map((product, index) => (
              <tr 
                key={`${product.stock_code}-${index}`}
                className="hover:bg-indigo-50/50 dark:hover:bg-indigo-900/20 transition-colors"
              >
                <td className="px-6 py-4">
                  <span className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-mono font-medium bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300">
                    {product.stock_code}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span className="text-sm font-medium text-slate-800 dark:text-slate-100">
                    {product.description}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span className="text-sm font-semibold text-emerald-600 dark:text-emerald-400">
                    £{product.unit_price.toFixed(2)}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span className="inline-flex items-center gap-1.5 text-sm text-slate-600 dark:text-slate-400">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    {product.country}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
