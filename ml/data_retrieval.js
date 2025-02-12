// // // const { models: { UserMovie, Movie, Friend } }  = require('../server/db') // Adjust the path if needed
// // // const _ = require('lodash');
// // // const { Op } = require('sequelize');
// // // const fs = require('fs');
// // // const path = require('path');




// // // async function fetchData() {
// // //   try {
// // //     // Fetch user-movie ratings
// // //     const userMovies = await UserMovie.findAll({
// // //       where: { rating: { [Op.not]: null } }, // Only include rows with ratings
// // //       raw: true,
// // //     });

// // //     // console.log('UserMovie:', userMovies);
// // //     // Fetch movie metadata
// // //     const movies = await Movie.findAll({
// // //       attributes: ['id', 'title', 'genres', 'criticScore', 'runtimeMinutes'],
// // //       raw: true,
// // //     });

// // //     // console.log('Movie:', movies);

// // //     // Fetch friend relationships
// // //     const friends = await Friend.findAll({
// // //       where: { status: 'accepted' },
// // //       raw: true,
// // //     });

// // //     console.log('Friend:', friends);

// // //     // Create a map for movies for quick lookup
// // //     const movieMap = _.keyBy(movies, 'id');

// // //     // Create a map for friends
// // //     const friendMap = _.groupBy(friends, 'userId');

// // //     // Add movie details and compute AvgFriendRating
// // //     const enrichedData = userMovies.map((entry) => {
// // //       const movie = movieMap[entry.movieId];
// // //       const userFriends = friendMap[entry.userId] || [];

// // //       // Get the IDs of the user's friends
// // //       const friendIds = userFriends.map((friend) => friend.friendId);

// // //       // Get the ratings from the user's friends for this movie
// // //       const friendRatings = userMovies
// // //         .filter((um) => friendIds.includes(um.userId) && um.movieId === entry.movieId)
// // //         .map((um) => um.rating);

// // //       const avgFriendRating = friendRatings.length
// // //         ? _.mean(friendRatings)
// // //         : null;

// // //       return {
// // //         ...entry,
// // //         title: movie?.title || 'Unknown',
// // //         genres: movie?.genres || [],
// // //         criticScore: movie?.criticScore || null,
// // //         runtimeMinutes: movie?.runtimeMinutes || null,
// // //         avgFriendRating,
// // //       };
// // //     });

// // //     // Save to CSV
// // //     const outputPath = path.join(__dirname, 'training_data.csv');
// // //     const csvData = [
// // //       'userId,movieId,rating,watched,dateWatched,title,genres,criticScore,runtimeMinutes,avgFriendRating',
// // //       ...enrichedData.map((row) =>
// // //         [
// // //           row.userId,
// // //           row.movieId,
// // //           row.rating,
// // //           row.watched,
// // //           row.dateWatched,
// // //           row.title,
// // //           row.genres.join('|'), // Join genres into a string
// // //           row.criticScore,
// // //           row.runtimeMinutes,
// // //           row.avgFriendRating,
// // //         ].join(',')
// // //       ),
// // //     ].join('\n');
// // //     fs.writeFileSync(outputPath, csvData);
// // //     console.log(`Data saved to ${outputPath}`);
// // //   } catch (error) {
// // //     console.error('Error fetching data:', error);
// // //   }
// // // }

// // // // Run the script
// // // fetchData();

// // const { models: { UserMovie, Movie, Friend } } = require('../server/db'); // Adjust path as needed
// // const _ = require('lodash');
// // const { Op } = require('sequelize');
// // const fs = require('fs');
// // const path = require('path');

// // // Function to fetch user, movie, and friend data
// // async function fetchData() {
// //   try {
// //     const userMovies = await UserMovie.findAll({
// //       where: { rating: { [Op.not]: null } }, // Only include rows with ratings
// //       raw: true,
// //     });

// //     const movies = await Movie.findAll({
// //       attributes: ['id', 'title', 'genres', 'criticScore', 'runtimeMinutes'],
// //       raw: true,
// //     });

// //     const friends = await Friend.findAll({
// //       where: { status: 'accepted' },
// //       raw: true,
// //     });

// //     return { userMovies, movies, friends };
// //   } catch (error) {
// //     console.error('Error fetching data:', error);
// //     throw error; // Propagate the error to the caller
// //   }
// // }

// // // Function to enrich the data
// // function enrichData(userMovies, movies, friends) {
// //   const movieMap = _.keyBy(movies, 'id');
// //   const friendMap = _.groupBy(friends, 'userId');

// //   return userMovies.map((entry) => {
// //     const movie = movieMap[entry.movieId];
// //     const userFriends = friendMap[entry.userId] || [];

// //     // Get friend IDs and their ratings for this movie
// //     const friendIds = userFriends.map((friend) => friend.friendId);
// //     const friendRatings = userMovies
// //       .filter((um) => friendIds.includes(um.userId) && um.movieId === entry.movieId)
// //       .map((um) => um.rating);

// //     const avgFriendRating = friendRatings.length ? _.mean(friendRatings) : null;

// //     return {
// //       ...entry,
// //       title: movie?.title || 'Unknown',
// //       genres: movie?.genres || [],
// //       criticScore: movie?.criticScore || null,
// //       runtimeMinutes: movie?.runtimeMinutes || null,
// //       avgFriendRating,
// //     };
// //   });
// // }

// // // Function to save enriched data to CSV
// // function saveToCSV(enrichedData) {
// //   try {
// //     const outputPath = path.join(__dirname, 'training_data.csv');
// //     const csvData = [
// //       'userId,movieId,rating,watched,dateWatched,title,genres,criticScore,runtimeMinutes,avgFriendRating',
// //       ...enrichedData.map((row) =>
// //         [
// //           row.userId,
// //           row.movieId,
// //           row.rating,
// //           row.watched,
// //           row.dateWatched,
// //           row.title,
// //           row.genres.join('|'), // Join genres into a string
// //           row.criticScore,
// //           row.runtimeMinutes,
// //           row.avgFriendRating,
// //         ].join(',')
// //       ),
// //     ].join('\n');

// //     fs.writeFileSync(outputPath, csvData);
// //     console.log(`Data saved to ${outputPath}`);
// //   } catch (error) {
// //     console.error('Error saving to CSV:', error);
// //   }
// // }

// // // Main function to fetch, enrich, and save data
// // async function main() {
// //   try {
// //     const { userMovies, movies, friends } = await fetchData();
// //     const enrichedData = enrichData(userMovies, movies, friends);
// //     saveToCSV(enrichedData);
// //   } catch (error) {
// //     console.error('Error in main function:', error);
// //   }
// // }

// // // Run the script
// // main();


// const { models: { UserMovie, Movie, Friend } } = require('../server/db'); // Adjust the path as needed
// const _ = require('lodash');
// const { Op } = require('sequelize');
// const fs = require('fs');
// const path = require('path');
// const axios = require('axios'); // To call the Flask API

// /**
//  * Fetch user, movie, and friend data from the database.
//  */
// async function fetchData() {
//   try {
//     const userMovies = await UserMovie.findAll({
//       where: { rating: { [Op.not]: null } },
//       raw: true,
//     });

//     const movies = await Movie.findAll({
//       attributes: ['id', 'title', 'criticScore', 'userRating'],
//       raw: true,
//     });

//     const friends = await Friend.findAll({
//       where: { status: 'accepted' },
//       raw: true,
//     });

//     return { userMovies, movies, friends };
//   } catch (error) {
//     console.error('Error fetching data from the database:', error);
//     throw error;
//   }
// }

// /**
//  * Enrich the user-movie data with movie details and average friend ratings.
//  */
// function enrichData(userMovies, movies, friends) {
//   try {
//     const movieMap = _.keyBy(movies, 'id');
//     const friendMap = _.groupBy(friends, 'userId');

//     return userMovies.map((entry) => {
//       const movie = movieMap[entry.movieId];
//       const userFriends = friendMap[entry.userId] || [];

//       const friendIds = userFriends.map((friend) => friend.friendId);
//       const friendRatings = userMovies
//         .filter((um) => friendIds.includes(um.userId) && um.movieId === entry.movieId)
//         .map((um) => um.rating);

//       const avgFriendRating = friendRatings.length ? _.mean(friendRatings) : null;

//       return {
//         ...entry,
//         title: movie?.title || 'Unknown',
//         criticScore: movie?.criticScore || null,
//         userRating: movie?.userRating || null,
//         avgFriendRating,
//       };
//     });
//   } catch (error) {
//     console.error('Error enriching data:', error);
//     throw error;
//   }
// }

// /**
//  * Save enriched data to a CSV file.
//  */
// function saveToCSV(data, filePath) {
//   try {
//     const csvHeader = 'userId,movieId,rating,status,dateWatched,title,criticScore,userRating,avgFriendRating';
//     const csvRows = data.map((row) =>
//       [
//         row.userId,
//         row.movieId,
//         row.rating,
//         row.status,
//         row.dateWatched,
//         row.title,
//         row.criticScore,
//         row.userRating,
//         row.avgFriendRating,
//       ].join(',')
//     );

//     const csvContent = [csvHeader, ...csvRows].join('\n');
//     fs.writeFileSync(filePath, csvContent);

//     console.log(`Data saved to ${filePath}`);
//   } catch (error) {
//     console.error('Error saving data to CSV:', error);
//     throw error;
//   }
// }

// /**
//  * Update predicted ratings for entries where status !== "watched".
//  */
// async function updateUnwatchedPredictions() {
//   try {
//     // Fetch UserMovie entries where status !== "watched"
//     const unwatchedMovies = await UserMovie.findAll({
//       where: { status: { [Op.not]: 'watched' } },
//       raw: true,
//     });

//     console.log(`Found ${unwatchedMovies.length} entries to update.`);

//     for (const userMovie of unwatchedMovies) {
//       const { userId, movieId } = userMovie;

//       try {
//         const response = await axios.post('http://127.0.0.1:5000/api/predict-rating', {
//           userId,
//           movieId,
//         });

//         const predictedRating = response.data.predictedRating;

//         await UserMovie.update(
//           { predictedRating },
//           {
//             where: { userId, movieId }, // Match the correct record
//           }
//         );

//         console.log(`Updated UserMovie(userId: ${userId}, movieId: ${movieId}) with predictedRating: ${predictedRating}`);
//       } catch (err) {
//         console.error(`Error predicting rating for UserMovie(userId: ${userId}, movieId: ${movieId}):`, err.message);
//       }
//     }

//     console.log('Finished updating predicted ratings.');
//   } catch (error) {
//     console.error('Error updating predictions:', error);
//   }
// }

// /**
//  * Main function to fetch, enrich, save, and update predictions.
//  */
// async function main() {
//   try {
//     console.log('Fetching data...');
//     const { userMovies, movies, friends } = await fetchData();

//     console.log('Enriching data...');
//     const enrichedData = enrichData(userMovies, movies, friends);

//     console.log('Saving data to CSV...');
//     const outputPath = path.join(__dirname, 'training_data.csv');
//     saveToCSV(enrichedData, outputPath);

//     console.log('Updating predicted ratings for entries with status !== "watched"...');
//     await updateUnwatchedPredictions();
//   } catch (error) {
//     console.error('Error in main function:', error);
//   }
// }

// // Run the script
// main();

/****************************************
 * data_retrieval.js
 ****************************************/
const { models: { UserMovie, Movie, Friend } } = require('../server/db'); // Adjust path as needed
const _ = require('lodash');
const { Op } = require('sequelize');
const fs = require('fs');
const path = require('path');
const axios = require('axios');

/**
 * Fetch user, movie, and friend data from the database.
 */
async function fetchData() {
  try {
    // 1) Fetch all UserMovie rows (including rating=null)
    const userMovies = await UserMovie.findAll({
      raw: true, // no filter
    });

    // 2) Fetch all Movies
    const movies = await Movie.findAll({
      attributes: ['id', 'title', 'criticScore', 'userRating'],
      raw: true,
    });

    // 3) Fetch accepted friend relationships
    const friends = await Friend.findAll({
      where: { status: 'accepted' },
      raw: true,
    });

    return { userMovies, movies, friends };
  } catch (error) {
    console.error('Error fetching data from the database:', error);
    throw error;
  }
}

/**
 * Enrich userMovie data with movie details (criticScore, userRating, etc.).
 * Optionally compute avgFriendRating (not strictly required for the CSV).
 */
function enrichData(userMovies, movies, friends) {
  try {
    const movieMap = _.keyBy(movies, 'id');
    const friendMap = _.groupBy(friends, 'userId');

    // Build rows from userMovies → merges movie data
    const dataRows = userMovies.map((entry) => {
      const movie = movieMap[entry.movieId];
      const userFriends = friendMap[entry.userId] || [];

      // Friend-based average rating if desired
      const friendIds = userFriends.map((f) => f.friendId);
      const friendRatings = userMovies
        .filter((um) => friendIds.includes(um.userId) && um.movieId === entry.movieId && um.rating != null)
        .map((um) => um.rating);

      const avgFriendRating = friendRatings.length ? _.mean(friendRatings) : null;

      return {
        userId: entry.userId,
        movieId: entry.movieId,
        rating: entry.rating,
        status: entry.status,
        dateWatched: entry.dateWatched,
        title: movie?.title || 'Unknown',
        criticScore: movie?.criticScore ?? null,
        userRating: movie?.userRating ?? null,
        avgFriendRating,
      };
    });

    // Optionally, if you want to include movies that have no userMovie row:
    // for (const movie of movies) {
    //   const found = dataRows.some((row) => row.movieId === movie.id);
    //   if (!found) {
    //     dataRows.push({
    //       userId: null,
    //       movieId: movie.id,
    //       rating: null,
    //       status: null,
    //       dateWatched: null,
    //       title: movie.title,
    //       criticScore: movie.criticScore,
    //       userRating: movie.userRating,
    //       avgFriendRating: null,
    //     });
    //   }
    // }

    return dataRows;
  } catch (error) {
    console.error('Error enriching data:', error);
    throw error;
  }
}

/**
 * Save enriched data to a CSV file.
 */
function saveToCSV(data, filePath) {
  try {
    const csvHeader = 'userId,movieId,rating,status,dateWatched,title,criticScore,userRating,avgFriendRating';
    const csvRows = data.map((row) =>
      [
        row.userId ?? '',
        row.movieId ?? '',
        row.rating ?? '',
        row.status ?? '',
        row.dateWatched ?? '',
        row.title ?? '',
        row.criticScore ?? '',
        row.userRating ?? '',
        row.avgFriendRating ?? '',
      ].join(',')
    );

    const csvContent = [csvHeader, ...csvRows].join('\n');
    fs.writeFileSync(filePath, csvContent);
    console.log(`Data saved to ${filePath}`);
  } catch (error) {
    console.error('Error saving data to CSV:', error);
    throw error;
  }
}

/**
 * Optionally update predicted ratings for entries where status !== "watched".
 */
async function updateUnwatchedPredictions() {
  try {
    const unwatchedMovies = await UserMovie.findAll({
      where: { status: { [Op.not]: 'watched' } },
      raw: true,
    });

    console.log(`Found ${unwatchedMovies.length} entries to update predicted rating.`);

    for (const userMovie of unwatchedMovies) {
      const { userId, movieId } = userMovie;
      try {
        const response = await axios.post('http://127.0.0.1:5000/api/predict-rating', {
          userId,
          movieId,
        });
        const predictedRating = response.data.predictedRating;

        await UserMovie.update(
          { predictedRating },
          { where: { userId, movieId } }
        );

        console.log(`Updated UserMovie(userId=${userId}, movieId=${movieId}) → predictedRating: ${predictedRating}`);
      } catch (err) {
        console.error(`Error predicting rating for userId=${userId}, movieId=${movieId}:`, err.message);
      }
    }

    console.log('Finished updating predicted ratings.');
  } catch (error) {
    console.error('Error updating predictions:', error);
  }
}

/**
 * Main driver function
 */
async function main() {
  try {
    console.log('Fetching data...');
    const { userMovies, movies, friends } = await fetchData();

    console.log('Enriching data...');
    const enrichedData = enrichData(userMovies, movies, friends);

    console.log('Saving data to CSV...');
    const outputPath = path.join(__dirname, 'training_data.csv');
    saveToCSV(enrichedData, outputPath);

    console.log('Optionally updating predicted ratings for unwatched entries...');
    await updateUnwatchedPredictions();
  } catch (error) {
    console.error('Error in main:', error);
  }
}

// Run the script
main();
