'use client'; // Add this at the top
import './globals.css';
import { Inter } from 'next/font/google';
import { AuthProvider } from './contexts/AuthContext';
import { useEffect } from 'react';

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  useEffect(() => {
    // Clean up VS Code injected classes
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