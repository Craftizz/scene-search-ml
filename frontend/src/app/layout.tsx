import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { VideoFramesProvider } from "../features/video/context/VideoFramesContext";

const inter = Inter({
  variable: "--font1",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SceneSearch",
  description: "A lightweight video search tool powered by machine learning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable}`}>
        <VideoFramesProvider>{children}</VideoFramesProvider>
      </body>
    </html>
  );
}
