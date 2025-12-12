"use client";
import { useState, FormEvent, ChangeEvent } from "react";
import styles from "./SearchBar.module.css";

type Props = {
	onSearch?: (q: string) => void;
};

export default function SearchBar({ onSearch }: Props) {
	const [query, setQuery] = useState("");

	function handleSubmit(e: FormEvent) {
		e.preventDefault();
		onSearch?.(query.trim());
	}

	function handleChange(e: ChangeEvent<HTMLInputElement>) {
		setQuery(e.target.value);
	}

	return (
		<form className={styles.searchbar} onSubmit={handleSubmit} role="search">
			<label htmlFor="search-input" className={styles.searchBarLabel}>Search</label>
			<input
				id="search-input"
				className={styles.searchBarInput}
				value={query}
				onChange={handleChange}
				placeholder="A person, place, or thing"
				aria-label="Search"
			/>
		</form>
	);
}
