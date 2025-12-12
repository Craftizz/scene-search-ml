import Image from "next/image";
import styles from "./page.module.css";
import Video from "@/features/video/components/Video";
import Search from "@/features/search/components/Search";
import Header from "@/components/header/Header";

export default function Home() {
  return (
    <div className={styles.container}>
      <Header />
      <div className={styles.content}>
        <Search />
        <Video />
      </div>
    </div>
  );
}
