"use client";

import { AuthProvider } from './contexts/AuthContext';
import { Inter } from 'next/font/google';
import './globals.css';
import { useEffect } from 'react';

const inter = Inter({ subsets: ['latin'] });

export default function ClientLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Remove any VS Code injected classes on mount
  useEffect(() => {
    document.body.className = inter.className;
  }, []);

  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}