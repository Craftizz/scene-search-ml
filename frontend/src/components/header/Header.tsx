"use client";
import Link from "next/link";
import styles from "./Header.module.css";

export default function Header() {
	return (
		<header className={styles.header}>
			<div className={styles.brand}>
				<h1 className={styles.logo}>SceneSearch</h1>
			</div>
		</header>
	);
}
